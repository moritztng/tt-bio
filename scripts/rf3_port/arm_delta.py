#!/usr/bin/env python3
"""Arm vs shipped arm at a matched seed: the structural move the arm itself causes.

X (device vs reference) is the number that goes next to the parity table, but at a rung
where the reference's own seed-to-seed spread R is large, X carries that spread as
irreducible noise and cannot separate two arms that differ by less than it. The arm-to-arm
distance does not: both sides replay the SAME recorded draws at the same seed, so the
diffusion trajectory is shared and everything that cancels, cancels. It is the metric the
two prior fold-level A2 verdicts used (CA 1.9335 A at 298 aa), so it is also the one that
compares to them.

It answers a different question from X -- "did the structure move" rather than "did it get
worse" -- so it qualifies X, it does not replace it.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from boltz2_affinity_parity import _kabsch_rmsd  # noqa: E402

R = REPO / "perf/rf3/results"
RUNGS = {"ubq_76": "ubq_76", "7roa_117": "7roa_117", "cdk2_128": "cdk2_128",
         "cdk2_298": "cdk2_298"}


def coords(d: Path, s: int, ca: bool):
    z = np.load(d / f"seed{s}.npz")
    return z["dev"][z["rep_idx"]] if ca else z["dev"]


def main() -> int:
    seeds = range(5)
    print("| rung | arm | CA arm-vs-shipped (A), per seed | mean | all-atom mean |")
    print("|---|---|---|--:|--:|")
    for rung in RUNGS:
        base = R / f"accuracy_{rung}"
        if not base.exists():
            continue
        for arm in ("a1", "a2", "a3"):
            d = R / f"accuracy_{rung}_{arm}"
            if not d.exists():
                continue
            ca = [_kabsch_rmsd(coords(base, s, True), coords(d, s, True)) for s in seeds]
            aa = [_kabsch_rmsd(coords(base, s, False), coords(d, s, False)) for s in seeds]
            print(f"| {rung} | {arm} | " + " ".join(f"{v:.4f}" for v in ca) +
                  f" | {np.mean(ca):.4f} | {np.mean(aa):.4f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
