#!/usr/bin/env python3
"""Agreement between `tt_bio.interface_scores` and the script Adaptyv scored Nipah with.

The test suite asserts the two agree to the last digit the reference prints. That is a
pass/fail; this prints the number behind it, per metric, over as many complexes as asked for:
the largest absolute difference, the mean, and the Pearson correlation of the two columns.

    python scripts/interface_scores_agreement.py --cases 60

Reference: `tests/fixtures/ipsae/nipah_ipsae_reference.py`, which is ipsae.py from
github.com/adaptyvbio/nipah_ipsae_pipeline @ 80e0d56, unmodified. Both implementations read the
same files, so this measures the definitions, not a fold. Nothing here claims agreement with a
GPU: a Tenstorrent fold and a CUDA fold are different coordinates, and that gap is a different
measurement on different inputs.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

from test_interface_scores import COLUMNS, _case, _reference  # noqa: E402

from tt_bio import interface_scores as isc  # noqa: E402


def collect(cases: int, tmp: Path, cutoffs: tuple[float, float]) -> dict[str, list[tuple[float, float]]]:
    """(reference, ours) for every metric over `cases` random complexes."""
    rng = np.random.default_rng(20260924)
    out: dict[str, list[tuple[float, float]]] = {key: [] for _, key, _ in COLUMNS}
    for i in range(cases):
        lengths = tuple(int(x) for x in rng.integers(12, 70, size=int(rng.integers(2, 4))))
        case_dir = tmp / f"c{i}"
        case_dir.mkdir(parents=True, exist_ok=True)
        d = _case(case_dir, i + 1000, lengths, ligand_atoms=int(rng.integers(0, 7)),
                  grid=bool(i % 3 == 0))
        ref = _reference(d, *cutoffs)
        ours = isc.score_files(d / "m_model_0.cif", d / "pae_m_model_0.npz",
                               d / "plddt_m_model_0.npz", d / "confidence_m_model_0.json", *cutoffs)
        for row in ref:
            a, b = row["Chn1"], row["Chn2"]
            got = (ours["directions"][f"{a}->{b}"] if row["Type"] == "asym"
                   else ours["pairs"]["-".join(sorted((a, b)))])
            for col, key, _ in COLUMNS:
                if got[key] is not None:
                    out[key].append((float(row[col]), float(got[key])))
    return out


def report(pairs: dict[str, list[tuple[float, float]]], cutoffs) -> int:
    print(f"tt_bio.interface_scores {isc.SCORES_VERSION} vs {isc.REFERENCE}")
    print(f"cutoffs {cutoffs[0]}/{cutoffs[1]}\n")
    print(f"{'metric':<12} {'n':>5} {'max |diff|':>12} {'mean |diff|':>12} {'pearson r':>11} "
          f"{'ref printed':>12}")
    worst = 0.0
    for col, key, dec in COLUMNS:
        v = np.asarray(pairs[key], dtype=float)
        if not len(v):
            continue
        diff = np.abs(v[:, 0] - v[:, 1])
        # The reference prints `dec` decimals, so a difference below half its last digit is the
        # printing, not the arithmetic. r is undefined when a column is constant; say so.
        r = (float(np.corrcoef(v[:, 0], v[:, 1])[0, 1])
             if v[:, 0].std() > 0 and v[:, 1].std() > 0 else float("nan"))
        worst = max(worst, float(diff.max()) - 0.5 * 10 ** -dec)
        print(f"{col:<12} {len(v):>5} {diff.max():>12.3e} {diff.mean():>12.3e} {r:>11.8f} "
              f"{0.5 * 10 ** -dec:>12.1e}")
    print(f"\nlargest excess over the reference's own printing precision: {worst:.3e}")
    return 0 if worst <= 1e-9 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=40)
    ap.add_argument("--pae_cutoff", type=float, default=isc.PAE_CUTOFF)
    ap.add_argument("--dist_cutoff", type=float, default=isc.DIST_CUTOFF)
    ap.add_argument("--tmp", default=None, help="Keep the generated complexes here.")
    a = ap.parse_args()
    cutoffs = (a.pae_cutoff, a.dist_cutoff)
    import tempfile
    with tempfile.TemporaryDirectory(prefix="ipsae-agree-") as td:
        root = Path(a.tmp) if a.tmp else Path(td)
        root.mkdir(parents=True, exist_ok=True)
        return report(collect(a.cases, root, cutoffs), cutoffs)


if __name__ == "__main__":
    sys.exit(main())
