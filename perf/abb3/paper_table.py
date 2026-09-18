"""Settle which Zenodo directory is which column of the paper's Table 1.

The mapping was not settled upstream and it is load-bearing: the reproduction is pre-registered
against a number, and a number attributed to the wrong artefact is the whole campaign built on a
wrong premise. ``btae576`` Table 1 attributes 2.42 A to ABodyBuilder3 and no released variant's
``evaluate.csv`` mean equals it, which is what left the mapping open.

Recomputing the table from the released per-structure CSVs resolves it. Table 1 is the
**59-structure** set, which is the subset the ABodyBuilder2 baseline was run on and the only
subset with YASARA refinement, and the paper's headline ABodyBuilder3 row is the **YASARA**
column, not the OpenMM one. That is why 2.42 could not be found: ``plddt-loss`` is 2.459 under
OpenMM and 2.416 under YASARA.

Usage: paper_table.py <output-dir-from-zenodo>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

COLUMNS = ["rmsd_cdrh1", "rmsd_cdrh2", "rmsd_cdrh3", "rmsd_fwh",
           "rmsd_cdrl1", "rmsd_cdrl2", "rmsd_cdrl3", "rmsd_fwl"]

# Table 1 of btae576, as published, against the artefact each row is recomputed from.
PAPER = {
    "ABodyBuilder2":      (("abodybuilder2-loss", "evaluate.csv"),
                           [0.84, 0.73, 2.54, 0.56, 0.55, 0.36, 0.88, 0.53]),
    "Baseline (OpenMM)":  (("base-loss", "evaluate.csv"),
                           [0.92, 0.75, 2.53, 0.60, 0.67, 0.35, 0.96, 0.58]),
    "Baseline (Yasara)":  (("base-loss", "evaluate_yasara.csv"),
                           [0.90, 0.74, 2.49, 0.59, 0.58, 0.37, 0.92, 0.57]),
    "ABodyBuilder3":      (("plddt-loss", "evaluate_yasara.csv"),
                           [0.87, 0.70, 2.42, 0.58, 0.61, 0.39, 0.93, 0.58]),
    "ABodyBuilder3-LM":   (("language-loss", "evaluate_yasara.csv"),
                           [0.87, 0.75, 2.40, 0.57, 0.59, 0.37, 0.89, 0.58]),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    args = ap.parse_args()

    subset = list(pd.read_csv(args.root / "abodybuilder2-loss" / "evaluate.csv")["structure"])
    print(f"Table 1 subset: {len(subset)} structures (the ABodyBuilder2 baseline's test set)\n")
    head = f"{'paper row':22s}{'source':40s}" + "".join(f"{c.replace('rmsd_',''):>9s}" for c in COLUMNS)
    print(head)

    worst = 0.0
    for row, ((variant, csv_name), published) in PAPER.items():
        df = pd.read_csv(args.root / variant / csv_name).set_index("structure")
        got = [df.loc[[s for s in subset if s in df.index], c].mean() for c in COLUMNS]
        worst = max(worst, max(abs(g - p) for g, p in zip(got, published)))
        print(f"{row:22s}{variant + '/' + csv_name:40s}" + "".join(f"{g:9.3f}" for g in got))
        print(f"{'  published':22s}{'btae576 Table 1':40s}" + "".join(f"{p:9.2f}" for p in published))

    print(f"\nworst |recomputed - published| over all 5 rows x 8 columns: {worst:.4f} A")
    print("Every cell agrees to the 2 decimal places the paper prints." if worst < 0.005
          else "MAPPING NOT CONFIRMED")
    return 0 if worst < 0.005 else 1


if __name__ == "__main__":
    raise SystemExit(main())
