#!/usr/bin/env python3
"""Turn a bond_coverage.py report into the evidence record stage_loss_coverage.py reads.

Derived, never typed: every number comes out of the measurement artifact, so a coverage
table cannot claim a share the run did not produce.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--term", default="bond")
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    r = json.loads(a.report.read_text())
    c = r["bond_gradient_contribution"]
    t = r["target"]
    rec = {
        "term": a.term,
        "stage": r["stage"],
        "dataset": r["dataset"],
        "target": f"{t['pdb_id']} {t['datapoint']}",
        "share_of_squared_gradient_norm": c["share_of_squared_gradient_norm"],
        "squared_norm_of_difference": c["squared_norm_of_difference"],
        "squared_norm_with_bond": c["squared_norm_with_bond"],
        "n_params_moved": c["n_params_moved"],
        "n_params": c["n_params"],
        "bond_mask_nnz": t["bond_mask_nnz"],
        "token_bonds_nnz": t["token_bonds_nnz"],
        "crop": r["crop"],
        "loss_with_term": r["arms"]["bond4"]["loss"],
        "loss_without_term": r["arms"]["bond0"]["loss"],
        "term_loss": r["arms"]["bond4"]["breakdown"].get("bond_loss"),
        "source": a.report.name,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
