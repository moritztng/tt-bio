#!/usr/bin/env python3
"""AF2-IG featurizer value-parity scorer (card-free, CPU-only).

`tt_bio.af2_data` rebuilds, without JAX, the feature dict ColabDesign hands AlphaFold2. The
reference is a committed capture of a real production forward pass
(`parity_artifacts/laczc128_b80/ref_inputs.npz`, taken through a `pre_callback` inside the jitted
model, so it is what AF2 actually consumed). This scorer rebuilds the same fixture and compares
every key bit-exact. Bit-exact is the right bar: the featurizer is integer and mask construction,
one-hots, and a coordinate copy.

The capture holds 78 arrays. 33 belong to the featurizer and are compared here. The other 45 are
ColabDesign design-loop state that a predict-only port never consumes, and they are enumerated
below rather than skipped silently -- a gate that only counts the keys it recognises cannot tell a
complete capture from a truncated one.

Report shape matches the rfd3 featurizer leg: `{verdict, keys_total, keys_bitexact, mismatches}`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
FIXTURE_DIR = REPO / "scripts" / "af2_port" / "parity_artifacts" / "laczc128_b80"
REF_NPZ = FIXTURE_DIR / "ref_inputs.npz"
TARGET_CIF = REPO / "perf" / "pxdesign" / "targets" / "laczc_128.cif"
BINDER_RESIDUES = 80

# The 33 keys tt_bio.af2_data owns.
FEATURE_KEYS = [
    "aatype", "target_feat", "msa_feat", "seq_mask", "msa_mask", "msa_row_mask",
    "atom14_atom_exists", "atom37_atom_exists",
    "residx_atom14_to_atom37", "residx_atom37_to_atom14", "residue_index",
    "extra_deletion_value", "extra_has_deletion", "extra_msa", "extra_msa_mask",
    "extra_msa_row_mask", "all_atom_positions",
    "template_aatype", "template_all_atom_mask", "template_all_atom_positions",
    "template_mask", "template_pseudo_beta", "template_pseudo_beta_mask",
    "asym_id", "sym_id", "entity_id",
    "batch/aatype", "batch/all_atom_positions", "batch/all_atom_mask",
    "rm_template", "rm_template_seq", "rm_template_sc", "mask_template_interchain",
]

# Everything else in the capture, and why the featurizer does not produce it.
NON_FEATURE_KEYS = {
    "_meta/json": "capture provenance",
    "prev/prev_pos": "recycle state, `initial_recycle_state` (checked below)",
    "prev/prev_pair": "recycle state, zero on the first pass",
    "prev/prev_msa_first_row": "recycle state, zero on the first pass",
    "use_dropout": "inference constant, False",
    "bias": "design-loop sequence bias, unused at predict time",
    "params/seq": "design-loop sequence logits; the port takes a sequence string",
    "seq/input": "design-loop sequence state",
    "seq/logits": "design-loop sequence state",
    "seq/pssm": "design-loop sequence state",
    "seq/soft": "design-loop sequence state",
    "seq/hard": "design-loop sequence state",
    "seq/pseudo": "design-loop sequence state",
}


def _pcc(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size == 0:
        return 1.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 1.0 if np.array_equal(a, b) else float("nan")
    return float((a * b).sum() / denom)


def featurizer_parity(work_dir: str | None = None) -> dict:
    """Rebuild the committed fixture and score every key against the capture."""
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "scripts" / "af2_port"))
    for path in (REF_NPZ, TARGET_CIF):
        if not path.exists():
            return {"mode": "af2ig_featurizer", "verdict": "ERROR",
                    "error": f"missing committed fixture {path}"}
    from af2_fixture import build_fixture

    from tt_bio.af2_data import complex_features, initial_recycle_state, monomer_features

    ref = np.load(REF_NPZ)
    work = Path(work_dir or "/tmp") / "af2ig_featurizer_parity"
    work.mkdir(parents=True, exist_ok=True)
    fixture = build_fixture(str(TARGET_CIF), str(work / "complex.pdb"), BINDER_RESIDUES)
    got = complex_features(str(work / "complex.pdb"), fixture["binder_seq"])

    mismatches: list[dict] = []
    bitexact = 0
    for key in FEATURE_KEYS:
        if key not in ref or key not in got:
            mismatches.append({"key": key, "reason": "MISSING",
                               "ported": key in got, "ref": key in ref})
            continue
        a, b = got[key], ref[key]
        if a.shape != b.shape:
            mismatches.append({"key": key, "reason": "SHAPE",
                               "ported": list(a.shape), "ref": list(b.shape)})
        elif a.dtype != b.dtype:
            mismatches.append({"key": key, "reason": "DTYPE",
                               "ported": str(a.dtype), "ref": str(b.dtype)})
        elif np.array_equal(a, b):
            bitexact += 1
        else:
            mismatches.append({"key": key, "reason": "VALUE", "pcc": _pcc(a, b),
                               "max_abs_diff": float(np.abs(
                                   a.astype(np.float64) - b.astype(np.float64)).max())})

    # The capture must be whole: every array is either a featurizer key or a named non-feature.
    accounted = set(FEATURE_KEYS) | set(NON_FEATURE_KEYS)
    unexpected = sorted(set(ref.files) - accounted - {k for k in ref.files
                                                      if k.startswith("opt/")})
    absent = sorted(accounted - set(ref.files))
    if unexpected or absent:
        mismatches.append({"key": "_capture_inventory", "reason": "INVENTORY",
                           "unexpected": unexpected, "absent": absent})

    prev = initial_recycle_state(got)
    prev_ok = {k: bool(np.array_equal(prev[k.split("/")[-1]], ref[k]))
               for k in ("prev/prev_pos", "prev/prev_pair", "prev/prev_msa_first_row")}
    if not all(prev_ok.values()):
        mismatches.append({"key": "_initial_recycle_state", "reason": "VALUE", **prev_ok})

    # The monomer stage takes no structure, so it has no capture to score against. What can be
    # checked card-free is that its sequence block is the complex's binder block verbatim: the
    # two stages must agree on the sequence or every RMSD between them is meaningless.
    monomer = monomer_features(fixture["binder_seq"])
    target_len = fixture["target_residues"]
    monomer_ok = (np.array_equal(monomer["aatype"], got["aatype"][target_len:])
                  and np.array_equal(monomer["msa_feat"], got["msa_feat"][:, target_len:])
                  and np.array_equal(monomer["residue_index"],
                                     np.arange(fixture["binder_residues"], dtype=np.int32))
                  and monomer["template_mask"].sum() == 0)
    if not monomer_ok:
        mismatches.append({"key": "_monomer_consistency", "reason": "VALUE"})

    report = {
        "mode": "af2ig_featurizer",
        "verdict": "PASS" if bitexact == len(FEATURE_KEYS) and not mismatches else "GAP",
        "keys_total": len(FEATURE_KEYS),
        "keys_bitexact": bitexact,
        "mismatches": mismatches,
        "fixture": {k: v for k, v in fixture.items() if k != "target_seq"},
        "capture_arrays": len(ref.files),
        "non_feature_arrays": len(ref.files) - len(FEATURE_KEYS),
    }
    return report


if __name__ == "__main__":
    _report = featurizer_parity()
    print(json.dumps(_report, indent=2, default=str))
    sys.exit(0 if _report["verdict"] == "PASS" else 1)
