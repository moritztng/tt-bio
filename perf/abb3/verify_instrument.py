"""Validate tt_bio/antibody_rmsd.py against ABodyBuilder3's own released per-structure numbers.

Recomputes every row of every released ``evaluate.csv`` from the released structures and reports
the max absolute deviation over all structures and all region columns. Nothing here may be tuned
to make our own model look better: the only thing it is fitted to is their structures returning
their published values.

Usage: verify_instrument.py <output-dir-from-zenodo> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

from tt_bio.antibody_rmsd import BACKBONE_ATOMS, TRUE_BACKBONE_ATOMS, score_pdb_pair

VARIANTS = ("base-loss", "plddt-loss", "language-loss", "abodybuilder2-loss")
BACKENDS = {"openmm": ("evaluate.csv", "refine"), "yasara": ("evaluate_yasara.csv", "refine_yasara")}
COLUMNS = [
    "rmsd_H", "rmsd_fwh", "rmsd_cdrh1", "rmsd_cdrh2", "rmsd_cdrh3",
    "rmsd_L", "rmsd_fwl", "rmsd_cdrl1", "rmsd_cdrl2", "rmsd_cdrl3",
]


def score_variant(root: Path, variant: str, pred_dir: str, structures, atoms):
    rows = []
    for name in structures:
        regions = list(torch.load(root / variant / "plddt" / f"{name}.pt", weights_only=False)["region"])
        scored = score_pdb_pair(
            root / variant / "true" / f"{name}.pdb",
            root / variant / pred_dir / f"{name}.pdb",
            regions,
            atoms,
        )
        rows.append({"structure": name, **scored})
    return pd.DataFrame(rows).set_index("structure")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    report: dict = {}
    worst = 0.0
    for variant in args.variants:
        for backend, (csv_name, pred_dir) in BACKENDS.items():
            csv = args.root / variant / csv_name
            if not csv.is_file() or not (args.root / variant / pred_dir).is_dir():
                continue
            theirs = pd.read_csv(csv).set_index("structure")
            ours = score_variant(args.root, variant, pred_dir, list(theirs.index), BACKBONE_ATOMS)
            delta = (ours[COLUMNS] - theirs[COLUMNS]).abs()
            key = f"{variant}/{backend}"
            per_col = {c: float(delta[c].max()) for c in COLUMNS}
            report[key] = {
                "n": int(len(ours)),
                "pred_dir": pred_dir,
                "max_abs_delta": float(delta.to_numpy().max()),
                "max_abs_delta_cdrh3": per_col["rmsd_cdrh3"],
                "per_column_max_abs_delta": per_col,
                "mean_cdrh3_ours": float(ours.rmsd_cdrh3.mean()),
                "mean_cdrh3_theirs": float(theirs.rmsd_cdrh3.mean()),
                "mean_fwh_ours": float(ours.rmsd_fwh.mean()),
                "mean_fwh_theirs": float(theirs.rmsd_fwh.mean()),
            }
            worst = max(worst, report[key]["max_abs_delta"])
            print(
                f"{key:28s} n={len(ours):3d} pred={pred_dir:14s} "
                f"max|delta|={report[key]['max_abs_delta']:.5f}  "
                f"CDR-H3 ours={report[key]['mean_cdrh3_ours']:.3f} "
                f"theirs={report[key]['mean_cdrh3_theirs']:.3f}"
            )

    # Control: the same code with the true backbone (N, CA, C, O) must NOT reproduce their
    # numbers. If it does, the agreement above is insensitive to the atom set and proves nothing.
    theirs = pd.read_csv(args.root / "base-loss" / "evaluate.csv").set_index("structure")
    ctrl = score_variant(args.root, "base-loss", "refine", list(theirs.index), TRUE_BACKBONE_ATOMS)
    ctrl_delta = float((ctrl[COLUMNS] - theirs[COLUMNS]).abs().to_numpy().max())
    report["control_N_CA_C_O"] = {
        "max_abs_delta": ctrl_delta,
        "mean_cdrh3_ours": float(ctrl.rmsd_cdrh3.mean()),
    }
    print(f"\ncontrol (N,CA,C,O backbone) max|delta|={ctrl_delta:.5f} "
          f"CDR-H3={ctrl.rmsd_cdrh3.mean():.3f} -- must be >> 0.01 or the check is vacuous")

    print(f"\nWORST max|delta| over everything: {worst:.5f}  (bar: <= 0.01)")
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return 0 if worst <= 0.01 and ctrl_delta > 0.01 else 1


if __name__ == "__main__":
    sys.exit(main())
