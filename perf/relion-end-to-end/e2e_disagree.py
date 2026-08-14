#!/usr/bin/env python3
"""How far apart are two RELION runs' answers, in degrees and Angstroms rather than in ==?

e2e_compare.py grades assignments with exact equality, which is the right test for two runs of the
SAME binary on the SAME box differing in one variable: there, bit-identical is achievable and the
count of non-identical particles is the signal.

It is the WRONG test the moment the two runs are different builds. RELION's own precalculated
Refine3D/job019 is its CUDA backend on someone else's machine; against our ALTCPU arm it scores
0/4452 identical on every column while the largest disagreement on _rlnAngleTilt over all 4452
particles is under a degree. Exact equality reads that as total disagreement. It is the opposite: the
runs agree everywhere and are bit-identical nowhere, which is what two float backends do.

So this reports the DISTRIBUTION of |delta| per column, with angles compared modulo 360 so that a
hair either side of the wrap does not score as 359.9 degrees of disagreement. The comparison the
parity verdict actually needs is between two of these: how far the bridge moves RELION's answer,
against how far RELION's own two backends already sit apart on the same data.

  python3 e2e_disagree.py ref_run tt_run gpu019_run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

S = Path("/home/ttuser/relion-scratch")
E2E = S / "e2e"
sys.path.insert(0, str(S))
from e2e_compare import star_col  # noqa: E402

ANG = ("_rlnAngleRot", "_rlnAngleTilt", "_rlnAnglePsi")
OFF = ("_rlnOriginXAngst", "_rlnOriginYAngst")


def col(stem, key):
    p = E2E / f"{stem}_data.star"
    if not p.exists():
        its = sorted(E2E.glob(f"{stem}_it0*_data.star"))
        if not its:
            return None
        p = its[-1]
    v = star_col(p, key)
    return np.array([float(x) for x in v]) if v else None


def delta(a, b, angular):
    d = np.abs(a - b)
    if angular:
        # Rot and Psi are periodic; a value at 359.99 and one at 0.01 are 0.02 degrees apart, not
        # 359.98. Tilt is not periodic in the same way but is bounded, so mod 360 is a no-op there.
        d = np.minimum(d, 360.0 - d)
    return d


def grade(a_stem, b_stem):
    out = {}
    print(f"\n=== {b_stem} against {a_stem} ===", flush=True)
    print(f"  {'column':22s} {'n':>5s} {'bit-identical':>14s} {'median':>10s} "
          f"{'p99':>10s} {'max':>10s}", flush=True)
    for c in ANG + OFF:
        a, b = col(a_stem, c), col(b_stem, c)
        if a is None or b is None or a.size != b.size:
            print(f"  {c:22s} unavailable", flush=True)
            continue
        d = delta(a, b, c in ANG)
        same = int((a == b).sum())
        out[c] = {"n": int(a.size), "identical": same, "median": float(np.median(d)),
                  "p99": float(np.percentile(d, 99)), "max": float(d.max())}
        unit = "deg" if c in ANG else "A"
        print(f"  {c:22s} {a.size:5d} {same:9d}/{a.size:<4d} "
              f"{np.median(d):10.6f} {np.percentile(d, 99):10.6f} {d.max():10.6f}  {unit}",
              flush=True)
    return out


def main():
    stems = sys.argv[1:] or ["ref_run", "tt_run", "gpu019_run"]
    base = stems[0]
    res = {"base": base, "pairs": {}}
    for s in stems[1:]:
        res["pairs"][s] = grade(base, s)
    (E2E / "e2e_disagree.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {E2E / 'e2e_disagree.json'}", flush=True)


if __name__ == "__main__":
    main()
