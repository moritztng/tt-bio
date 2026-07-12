#!/usr/bin/env python3
"""Reshape an upstream `boltz predict` output dir into tt-bio's results.json/structures/
layout, so scripts/pharma_parity.py's model-agnostic `structures` mode (built around
tt-bio predict's own output) can drive the Boltz-2 leg without a second statistical core.

Usage: boltz2_ref_layout.py REF_DIR OUT_DIR [--model model_0]
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def convert(ref_dir: Path, out_dir: Path, model: str = "model_0") -> list:
    pred_root = ref_dir / "predictions"
    struct_out = out_dir / "structures"
    struct_out.mkdir(parents=True, exist_ok=True)
    records = []
    for tid_dir in sorted(pred_root.iterdir()):
        tid = tid_dir.name
        cif = tid_dir / f"{tid}_{model}.cif"
        conf = tid_dir / f"confidence_{tid}_{model}.json"
        if not cif.exists() or not conf.exists():
            continue
        shutil.copy(cif, struct_out / f"{tid}.cif")
        rec = json.loads(conf.read_text())
        rec["id"] = tid
        records.append(rec)
    (out_dir / "results.json").write_text(json.dumps(records, indent=2))
    return records


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="model_0")
    args = ap.parse_args()
    recs = convert(Path(args.ref_dir), Path(args.out_dir), args.model)
    print(f"converted {len(recs)} targets -> {args.out_dir}")
