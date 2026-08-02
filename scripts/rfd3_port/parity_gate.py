#!/usr/bin/env python3
"""Gate-ready RFD3 featurizer value-parity scorer (card-free, CPU-only).

The RFD3 port's correctness anchor is value parity of the host featurizer
(``tt_bio.rfd3.featurize.featurize``) against a captured reference `f` from
the upstream RosettaCommons/foundry featurizer. The reference is COMMITTED
(``scripts/rfd3_port/parity_artifacts/iai_protein/ref_f.pt``), so this scorer
runs the ported featurizer on the committed ``IAI_protein.pdb`` + contig and
compares every `f` key bit-exact — no foundry install and no device needed at
gate time. This is the same 43/43-key bit-exactness verified during the port
(p12, protein-binder/motif-scaffold case F1/F6); the gate re-runs it every
release so a featurizer regression can't ship silently.

Returns a report dict consumed by ``scripts/full_parity_gate.py`` (mode ==
"rfd3_featurizer"): ``{verdict, keys_total, keys_bitexact, mismatches, ...}``.
PASS iff every comparable key is bit-exact (the port's own bar, per
``scripts/rfd3_port/parity_artifacts/README.md``).

This mirrors the in-process leg pattern (esmc/saprot/esmfold2): no fixture dir,
no device fold, just a direct in-process check that writes its report to the
gate's workdir for --resume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
ARTIFACTS = REPO / "scripts" / "rfd3_port" / "parity_artifacts"
IAI_DIR = ARTIFACTS / "iai_protein"
PDB = IAI_DIR / "IAI_protein.pdb"
REF_PT = IAI_DIR / "ref_f.pt"
CONTIG = "A1-10,20,A31-40"

# The 43 keys the port verified bit-exact against the reference (p12): 19
# token-level + 24 atom-level. See parity_artifacts/README.md for the per-key
# semantics. Order is stable for the mismatch report.
TOKEN_KEYS = [
    "restype", "ref_motif_token_type", "ref_plddt", "is_non_loopy",
    "is_motif_token_unindexed", "is_motif_token_with_fully_fixed_coord",
    "is_protein", "is_rna", "is_dna", "is_ligand", "is_polar",
    "terminus_type", "asym_id", "entity_id", "sym_id",
    "residue_index", "token_index", "token_bonds", "unindexing_pair_mask",
]
ATOM_KEYS = [
    "ref_atom_name_chars", "ref_pos", "ref_mask", "ref_element", "ref_charge",
    "ref_space_uid", "ref_pos_is_ground_truth", "has_zero_occupancy",
    "ref_is_motif_atom_with_fixed_coord", "ref_is_motif_atom_unindexed",
    "ref_atomwise_rasa", "active_donor", "active_acceptor", "is_atom_level_hotspot",
    "is_motif_atom_with_fixed_coord", "is_motif_atom_with_fixed_seq",
    "is_motif_atom_unindexed", "motif_pos", "is_ca", "is_central",
    "is_backbone", "is_sidechain", "is_virtual", "atom_to_token_map",
]
ALL_KEYS = TOKEN_KEYS + ATOM_KEYS


def _pcc(a, b) -> float:
    import torch
    a = a.float().flatten(); b = b.float().flatten()
    if a.numel() == 0:
        return 1.0
    a = a - a.mean(); b = b - b.mean()
    denom = a.norm() * b.norm()
    if denom == 0:
        return 1.0 if torch.equal(a, b) else float("nan")
    return float((a * b).sum() / denom)


def featurizer_parity() -> dict:
    """Run the ported featurizer on the committed IAI fixture and compare every
    `f` key bit-exact vs the committed reference capture. Card-free, CPU-only."""
    import torch
    sys.path.insert(0, str(REPO))
    from tt_bio.rfd3.featurize import featurize
    from tt_bio.rfd3.input import InputSpecification

    if not PDB.exists():
        return {"mode": "rfd3_featurizer", "verdict": "ERROR",
                "error": f"missing committed fixture {PDB}"}
    if not REF_PT.exists():
        return {"mode": "rfd3_featurizer", "verdict": "ERROR",
                "error": f"missing committed reference {REF_PT}"}

    spec = InputSpecification.from_dict({"input": str(PDB), "contig": CONTIG})
    pf = featurize(str(PDB), spec)
    rf = torch.load(REF_PT, weights_only=False)

    keys_total = 0
    keys_bitexact = 0
    mismatches: list[dict] = []
    for k in ALL_KEYS:
        if k not in pf or k not in rf:
            keys_total += 1
            mismatches.append({"key": k, "reason": "MISSING",
                               "ported": k in pf, "ref": k in rf})
            continue
        keys_total += 1
        a, b = pf[k], rf[k]
        if a.shape != b.shape:
            mismatches.append({"key": k, "reason": "SHAPE",
                                "ported": list(a.shape), "ref": list(b.shape)})
            continue
        if torch.equal(a, b):
            keys_bitexact += 1
        else:
            mismatches.append({"key": k, "reason": "VALUE",
                                "pcc": _pcc(a, b)})

    verdict = "PASS" if keys_bitexact == keys_total and not mismatches else "GAP"
    return {
        "mode": "rfd3_featurizer",
        "verdict": verdict,
        "keys_total": keys_total,
        "keys_bitexact": keys_bitexact,
        "mismatches": mismatches,
        "fixture": str(IAI_DIR.relative_to(REPO)),
        "contig": CONTIG,
        "ported_tokens": int(pf["restype"].shape[0]),
        "ported_atoms": int(pf["ref_pos"].shape[0]),
        "ref_tokens": int(rf["restype"].shape[0]),
        "ref_atoms": int(rf["ref_pos"].shape[0]),
    }


def main() -> int:
    rep = featurizer_parity()
    print(json.dumps(rep, indent=2))
    return 0 if rep.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
