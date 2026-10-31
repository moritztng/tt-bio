"""Protenix / Mol-1 MSA features built exactly as upstream builds them at inference.

``upstream_msa_features`` is ``mol.data.msa.msa_featurizer.InferenceMSAFeaturizer.make_msa_feature`` with the
AtomArray replaced by what tt-bio's featurizer already knows (per-token asym index and residue number, per-chain
atom count). The assembly itself, deduplication, species pairing, the 600-per-species and 16384-row caps and the
paired-rows-also-unpaired merge, is the vendored upstream code in ``tt_bio._vendor.mol1``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from tt_bio._vendor.mol1.constants import DNA_CHAIN, PROTEIN_CHAIN, RNA_CHAIN
from tt_bio._vendor.mol1.msa_assembly import FeatureAssemblyLine, ensure_ends_with_newline
from tt_bio._vendor.mol1.msa_utils import RawMsa, map_to_standard

_CTYPE = {"protein": PROTEIN_CHAIN, "rna": RNA_CHAIN, "dna": DNA_CHAIN}
LIGAND = "ligand"


@dataclass
class ChainMSA:
    """One chain (one copy) in token order. ``entity`` groups copies; ``n_atoms`` matters only for a ligand,
    whose upstream "sequence" is one X per atom."""
    chain_id: str
    entity: int
    mol_type: str                 # protein | dna | rna | ligand
    sequence: str
    n_atoms: int = 0
    paired: str | None = None
    unpaired: str | None = None


def upstream_msa_features(chains: list[ChainMSA], token_asym: torch.Tensor, token_res: torch.Tensor,
                          msa_pair_as_unpair: bool = True, use_rna_msa: bool = False) -> dict:
    """Per-token MSA features (msa, has_deletion, deletion_value, profile, deletion_mean) for the whole complex.

    ``token_asym`` is each token's chain position in ``chains`` (0-based) and ``token_res`` its residue number
    (1-based; every atom token of a ligand component carries its component number), as upstream reads them off the
    token centre atoms."""
    meta, merged = {}, {}
    for aid, c in enumerate(chains):
        ctype = _CTYPE.get(c.mol_type, LIGAND)
        seq = "X" * c.n_atoms if ctype == LIGAND else c.sequence
        p = ensure_ends_with_newline(c.paired) if ctype == PROTEIN_CHAIN else None
        u = c.unpaired if ctype == PROTEIN_CHAIN or (ctype == RNA_CHAIN and use_rna_msa) else None
        u = ensure_ends_with_newline(u)
        if msa_pair_as_unpair and p:
            # upstream merges once per entity; copies of one entity share the same text, so per chain is the same
            key = (c.entity, p, u)
            if key not in merged:
                merged[key] = RawMsa.from_a3m(seq, ctype, p + (u or ""), dedup=True).to_a3m()
            u = merged[key]
        meta[aid] = {"entity_id": c.entity, "chain_id": c.chain_id, "sequence": seq,
                     "paired_msa": p or "", "unpaired_msa": u or "",
                     "chain_entity_type": ctype if ctype != LIGAND else "non-polymer"}
    std = map_to_standard(np.asarray(token_asym), np.asarray(token_res), meta)
    f = FeatureAssemblyLine().assemble(meta, std).to_dict()
    return {"msa": torch.from_numpy(f["msa"].astype(np.int64)),
            "has_deletion": torch.from_numpy(f["has_deletion"].astype(np.float32)),
            "deletion_value": torch.from_numpy(f["deletion_value"].astype(np.float32)),
            "profile": torch.from_numpy(f["profile"].astype(np.float32)),
            "deletion_mean": torch.from_numpy(f["deletion_mean"].astype(np.float32))}
