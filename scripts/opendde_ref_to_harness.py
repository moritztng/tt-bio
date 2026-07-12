#!/usr/bin/env python3
"""Convert an OpenDDE reference CLI (opendde pred) output tree into the
pharma-parity harness layout: a directory with results.json and structures/<id>.cif.

The reference CLI writes <out>/<name>/seed_<s>/predictions/<name>_sample_<k>.cif
plus a per-sample <name>_summary_confidence_sample_<k>.json. The harness
(scripts/pharma_parity.py structures) expects one dir per seed with a single
confidence-selected cif at structures/<id>.cif and a results.json list whose
entries carry the id and the confidence keys it shares with the device path.

Confidence-key mapping (reference summary -> device results.json key), faithful
subset only; unmapped harness keys are omitted so the harness prints an honest
dash rather than a fabricated value:

    ranking_score  -> confidence_score
    ptm            -> ptm
    iptm           -> iptm
    plddt / 100    -> complex_plddt   (reference pLDDT is 0-100, device is 0-1)

The parity verdict (R/D/X Kabsch RMSD, coord PCC) is computed from cif
coordinates only, so it is unaffected by the confidence-key subset.

Usage:
    python3 scripts/opendde_ref_to_harness.py <ref_pred_out> <name> <seed> <harness_dir>
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    ref_out, name, seed, harness_dir = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    pred = ref_out / name / f"seed_{seed}" / "predictions"
    if not pred.is_dir():
        print(f"no predictions dir at {pred}", file=sys.stderr)
        return 1
    sums = sorted(pred.glob(f"{name}_summary_confidence_sample_*.json"))
    if not sums:
        print(f"no summary json in {pred}", file=sys.stderr)
        return 1
    best_k, best_rs = None, -1.0
    for s in sums:
        d = json.load(open(s))
        if d.get("ranking_score", -1.0) > best_rs:
            best_rs = d["ranking_score"]; best_k = int(s.stem.rsplit("_", 1)[-1])
    cif_src = pred / f"{name}_sample_{best_k}.cif"
    st = harness_dir / "structures"; st.mkdir(parents=True, exist_ok=True)
    shutil.copy(cif_src, st / f"{name}.cif")
    d = json.load(open(pred / f"{name}_summary_confidence_sample_{best_k}.json"))
    entry = {"id": name, "status": "ok",
             "confidence_score": d["ranking_score"], "ptm": d["ptm"], "iptm": d["iptm"],
             "complex_plddt": d["plddt"] / 100.0}
    (harness_dir / "results.json").write_text(json.dumps([entry], indent=2))
    print(f"{harness_dir}: structures/{name}.cif + results.json (sample {best_k}, rs={best_rs:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
