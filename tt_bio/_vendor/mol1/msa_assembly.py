# Vendored from Mol-1 (https://github.com/molence/Mol-1) @ e6e64866ee89640a91cef7ee777a85f1d8be03bb, Apache-2.0.
# Copyright 2024 ByteDance and/or its affiliates; modifications Copyright 2026 Molence Bio.
# Regenerate with scripts/mol1_io/vendor_msa.py; do not edit by hand.
import dataclasses
from typing import Any, Dict, Mapping, Optional

import numpy as np

from .constants import PROTEIN_CHAIN, RNA_CHAIN, STANDARD_POLYMER_CHAIN_TYPES, STD_RESIDUES_WITH_GAP
from .msa_utils import MSA_GAP_IDX, MSAPairingEngine, NUM_SEQ_NUM_RES_MSA_FEATURES, RawMsa, map_to_standard


class FeatureAssemblyLine:
    """
    Orchestrates the conversion of Raw MSAs into finalized Mol-1 features.

    Args:
        max_msa_size: Maximum number of sequences allowed in the final MSA.
        max_paired_per_species: Maximum number of paired sequences per species.
    """

    def __init__(
        self, max_msa_size: int = 16384, max_paired_per_species: int = 600
    ) -> None:
        self.max_size = max_msa_size
        self.max_paired_per_sp = max_paired_per_species

    def assemble(
        self, bioassembly: Mapping[int, Mapping[str, Any]], std_idxs: np.ndarray
    ) -> "MSAFeat":
        """
        Executes the complete feature assembly pipeline.

        Args:
            bioassembly: Mapping of asymmetric IDs to chain information.
            std_idxs: Array of standardized residue indices.

        Returns:
            An assembled MSAFeat object.
        """
        # 1. Base featurization
        unique_prot_seqs = {
            v["sequence"]
            for v in bioassembly.values()
            if v["chain_entity_type"] == PROTEIN_CHAIN
        }
        need_pairing = len(unique_prot_seqs) > 1
        active_chain_ids = {v["chain_id"] for v in bioassembly.values()}

        raw_chains = []
        for aid, info in bioassembly.items():
            ctype, seq = info["chain_entity_type"], info["sequence"]
            skip = ctype not in STANDARD_POLYMER_CHAIN_TYPES or len(seq) <= 4

            if ctype in STANDARD_POLYMER_CHAIN_TYPES:
                up_msa = RawMsa.from_a3m(
                    seq,
                    ctype,
                    (
                        info["unpaired_msa"]
                        if not skip and ctype in [PROTEIN_CHAIN, RNA_CHAIN]
                        else ""
                    ),
                    dedup=True,
                )
                p_msa = RawMsa.from_a3m(
                    seq,
                    ctype,
                    (
                        info["paired_msa"]
                        if not skip and need_pairing and ctype == PROTEIN_CHAIN
                        else ""
                    ),
                    dedup=False,
                )
            else:
                up_msa = p_msa = RawMsa(
                    seq, PROTEIN_CHAIN, [], [], deduplicate=False
                )  # Ligand placeholders

            u_f, p_f = up_msa.featurize(), p_msa.featurize()
            chain_feat = dict(u_f)
            chain_feat.update({f"{k}_all_seq": v for k, v in p_f.items()})
            chain_feat.update(
                {
                    "asym_id": np.full(len(seq), aid),
                    "chain_id": info["chain_id"],
                    "entity_id": info["entity_id"],
                }
            )
            # Compute Profile
            msa = chain_feat["msa"]
            prof = (msa[..., None] == np.arange(len(STD_RESIDUES_WITH_GAP))).sum(
                axis=0
            ) / msa.shape[0]
            chain_feat.update(
                {
                    "profile": prof.astype(np.float32),
                    "deletion_mean": np.mean(chain_feat["deletion_matrix"], axis=0),
                }
            )
            raw_chains.append(chain_feat)

        # 2. Pairing and cleanup
        max_p = self.max_size // 2
        if need_pairing:
            raw_chains = MSAPairingEngine.pair_chains_by_species(
                raw_chains, max_p, active_chain_ids, self.max_paired_per_sp
            )
            raw_chains = MSAPairingEngine.cleanup_unpaired_features(raw_chains)

        # 3. Filter all-gap rows
        nonempty_asyms = [
            c["asym_id"][0] for c in raw_chains if c["chain_id"] in active_chain_ids
        ]
        if "msa_all_seq" in raw_chains[0]:
            raw_chains = MSAPairingEngine.filter_all_gapped_rows(
                raw_chains, nonempty_asyms
            )

        # 4. Cropping and merging
        cropped = []
        for c in raw_chains:
            p_msa = c.get("msa_all_seq")
            ps = min(p_msa.shape[0], max_p) if p_msa is not None else 0
            us = max(0, min(c["msa"].shape[0], self.max_size - ps))

            cr = {
                "asym_id": c["asym_id"],
                "chain_id": c["chain_id"],
                "profile": c["profile"],
                "deletion_mean": c["deletion_mean"],
            }
            for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
                if k in c:
                    cr[k] = c[k][:us]
                if f"{k}_all_seq" in c:
                    cr[f"{k}_all_seq"] = c[f"{k}_all_seq"][:ps]
            cropped.append(cr)

        merged = {"asym_id": np.concatenate([c["asym_id"] for c in cropped])}
        for base in NUM_SEQ_NUM_RES_MSA_FEATURES:
            for f in [base, f"{base}_all_seq"]:
                if f in cropped[0]:
                    merged[f] = MSAPairingEngine.merge_chain_features(cropped, f)
        for f in ["profile", "deletion_mean"]:
            merged[f] = np.concatenate([c[f] for c in cropped])

        # 5. Depth tracking
        active_set = set(nonempty_asyms)
        max_u = max([len(c["msa"]) for c in cropped if c["asym_id"][0] in active_set])
        rna_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == RNA_CHAIN
            ]
        )
        prot_u = max(
            [1]
            + [
                len(c["msa"])
                for c in cropped
                if bioassembly[c["asym_id"][0]]["chain_entity_type"] == PROTEIN_CHAIN
            ]
        )

        merged["msa"] = merged["msa"][:max_u]
        prot_p = 1
        if "msa_all_seq" in merged:
            max_p_actual = max(
                [
                    len(c["msa_all_seq"])
                    for c in cropped
                    if c["asym_id"][0] in active_set
                ]
            )
            merged["msa_all_seq"] = merged["msa_all_seq"][:max_p_actual]
            prot_p = max_p_actual

        # 6. Final integration and coordinate mapping
        for k in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if k in merged and f"{k}_all_seq" in merged:
                merged[k] = np.concatenate([merged[f"{k}_all_seq"], merged[k]], axis=0)

        # Forward compatibility patch for non-protein entities
        for aid in [
            aid
            for aid, info in bioassembly.items()
            if info["chain_entity_type"] != PROTEIN_CHAIN
        ]:
            cols = np.where(merged["asym_id"] == aid)[0]
            if cols.size > 0:
                gap_mask = np.all(merged["msa"][:, cols] == MSA_GAP_IDX, axis=1)
                merged["msa"][np.ix_(np.where(gap_mask)[0], cols)] = merged["msa"][
                    0, cols
                ]

        for f in NUM_SEQ_NUM_RES_MSA_FEATURES:
            if f in merged:
                merged[f] = merged[f][:, std_idxs].copy()
        for f in ["profile", "deletion_mean"]:
            merged[f] = merged[f][std_idxs]

        def to_i8(x: np.ndarray) -> np.ndarray:
            return np.clip(x, -128, 127).astype(np.int8)

        return MSAFeat(
            rows=to_i8(merged["msa"]),
            mask=np.ones_like(merged["msa"], dtype=bool),
            deletion_matrix=to_i8(merged["deletion_matrix"]),
            profile=merged["profile"],
            deletion_mean=merged["deletion_mean"],
            prot_unpaired_num_alignments=np.array(prot_u, dtype=np.int32),
            prot_paired_num_alignments=np.array(prot_p, dtype=np.int32),
            rna_unpaired_num_alignments=np.array(rna_u, dtype=np.int32),
        )

@dataclasses.dataclass(frozen=True)
class MSAFeat:
    """Container for finalized numerical MSA features."""

    rows: np.ndarray
    mask: np.ndarray
    deletion_matrix: np.ndarray
    profile: np.ndarray
    deletion_mean: np.ndarray
    prot_unpaired_num_alignments: np.ndarray
    prot_paired_num_alignments: np.ndarray
    rna_unpaired_num_alignments: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        """Converts the MSA object into a standard Mol-1 data dictionary."""
        return {
            "msa": self.rows,
            "msa_mask": self.mask,
            "deletion_matrix": self.deletion_matrix,
            "deletion_value": (np.arctan(self.deletion_matrix / 3.0) * (2.0 / np.pi)),
            "has_deletion": np.clip(self.deletion_matrix, 0.0, 1.0),
            "profile": self.profile,
            "deletion_mean": self.deletion_mean,
            "prot_unpaired_num_alignments": self.prot_unpaired_num_alignments,
            "prot_paired_num_alignments": self.prot_paired_num_alignments,
            "rna_unpaired_num_alignments": self.rna_unpaired_num_alignments,
            "rna_pair_num_alignments": np.asarray(1, dtype=np.int32),
            "prot_pair_num_alignments": self.prot_paired_num_alignments,
            "prot_unpair_num_alignments": self.prot_unpaired_num_alignments,
            "rna_unpair_num_alignments": self.rna_unpaired_num_alignments,
        }


def ensure_ends_with_newline(s: Optional[str]) -> Optional[str]:
    """
    Ensure the given string ends with a newline character.

    If the string is non-empty and does not already end with '\\n',
    append '\\n'. Empty strings are returned unchanged.

    Args:
        s (str): Input string.

    Returns:
        str: The input string guaranteed to end with '\\n' when non-empty.
    """
    if not s:
        return s
    if not s.endswith("\n"):
        s += "\n"
    return s

