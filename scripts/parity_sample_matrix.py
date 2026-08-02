#!/usr/bin/env python3
"""Pairwise CA-Kabsch matrix between the samples of two or more multi-sample folds.

The parity gate scores one structure per leg: the confidence-rank-0 sample, ``<tid>.cif``. When a
model's top confidences are tied more tightly than the arithmetic difference under test, which
sample lands there is not a stable observable, and a rank-0-to-rank-0 comparison reports the
distance between two DIFFERENT samples as a port error. This prints the full matrix so the
reordering is visible: a run that reproduced the reference has one near-zero entry per row, off the
diagonal when the ranking permuted.

That is how ``protenix-v2-prot-msa`` was root-caused on 2026-07-31 (2.403 A reported, 0.139 A real
— see docs/implementation-parity.md). ``scripts/integration_envelope.py`` now matches samples
before scoring; this stays the diagnostic you reach for when a structure leg GAPs, before hunting
a precision boundary.

    scripts/parity_sample_matrix.py prot \
        dev=/tmp/gate/protenix-prot-msa/seed0/protenix_results_prot/structures \
        fp32=.../ref_fp32/protenix_results_prot/structures
"""
from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from integration_envelope import _ca_coords, _pair_rmsd  # noqa: E402


def _samples(d: Path, tid: str):
    """Rank-ordered sample CIFs: rank 0 is <tid>.cif, ranks 1.. are <tid>_model_<rank>.cif."""
    out = [d / f"{tid}.cif"]
    r = 1
    while (d / f"{tid}_model_{r}.cif").exists():
        out.append(d / f"{tid}_model_{r}.cif")
        r += 1
    return out


def main() -> int:
    if len(sys.argv) < 4 or "=" not in sys.argv[2]:
        print(__doc__)
        return 2
    tid = sys.argv[1]
    runs = {}
    for spec in sys.argv[2:]:
        label, path = spec.split("=", 1)
        runs[label] = [_ca_coords(p) for p in _samples(Path(path), tid)]

    def rmsd(a, b):
        v = _pair_rmsd(a, b)
        return float("nan") if v is None else v

    print("### within-run spread (rank 0 vs rank k)")
    for label, s in runs.items():
        print(f"{label:12s} " + " ".join(f"{rmsd(s[0], x):7.3f}" for x in s))
    for l1, l2 in itertools.combinations(runs, 2):
        a, b = runs[l1], runs[l2]
        print(f"\n### {l1} rank-k (rows) vs {l2} rank-j (cols)")
        print(" " * 12 + " ".join(f"{'r' + str(j):>7s}" for j in range(len(b))))
        for i, ai in enumerate(a):
            row = [rmsd(ai, bj) for bj in b]
            best = int(np.nanargmin(row))
            print(f"r{i:<11d}" + " ".join(f"{v:7.3f}" for v in row) + f"   <- best r{best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
