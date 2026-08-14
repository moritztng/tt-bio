#!/usr/bin/env python3
"""Fit each arm's wall against particle count, then compose the production-scale verdict table.

Every arm here ran the SAME iteration (`--continue` from RELION's own it012 optimiser,
`--auto_iter_max 13`) at 4,452 / 13,356 / 35,616 / 111,300 particles, so a wall is
`fixed + marginal x N` and the two terms answer different questions:

  fixed     work that does not scale with particle count -- startup, the preread, the optimiser
            read, the M-step. This is what a bigger job amortises, and amortising it is the entire
            mechanism behind `relion-intercard-scaling`'s ~7,060-particle crossover.
  marginal  the per-particle cost. This is the only term that can be compared between two machines
            without the job size smuggling itself into the ratio.

`relion-end-to-end` §7.3's comparison was built from one job size, so it could not separate them.

Reads the `.time` files the campaigns wrote (`/usr/bin/time -f "%e %U %S %M"`). Missing arms are
reported as missing rather than filled in.
"""
import glob
import os
import sys

SCALES = {"x1": 4452, "x3": 13356, "x8": 35616, "x25": 111300}


def load(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "*.time"))):
        arm = os.path.basename(p)[:-5]
        txt = open(p).read().split()
        if len(txt) < 4:
            continue
        base, _, sc = arm.rpartition("_")
        if sc not in SCALES:
            continue
        out.setdefault(base, {})[SCALES[sc]] = float(txt[0])
    return out


def fit(pts):
    """Least squares wall = a + b*N. Returns (a, b, worst residual %)."""
    n = len(pts)
    if n < 2:
        return None
    xs = sorted(pts)
    sx = sum(xs); sy = sum(pts[x] for x in xs)
    sxx = sum(x * x for x in xs); sxy = sum(x * pts[x] for x in xs)
    b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
    a = (sy - b * sx) / n
    worst = max(abs(a + b * x - pts[x]) / pts[x] * 100 for x in xs)
    return a, b, worst


def main():
    dirs = sys.argv[1:] or ["gpu", "qb1"]
    arms = {}
    for d in dirs:
        if os.path.isdir(d):
            arms.update(load(d))
        else:
            print(f"# missing directory {d}")
    print(f"{'arm':10s} {'points':>28s} {'fixed s':>9s} {'ms/particle':>12s} {'worst resid':>12s}")
    fits = {}
    for arm in sorted(arms):
        pts = arms[arm]
        cells = " ".join(f"{n//1000}k:{pts[n]:.1f}" for n in sorted(pts))
        f = fit(pts)
        fits[arm] = f
        if f:
            print(f"{arm:10s} {cells:>28s} {f[0]:9.2f} {f[1]*1e3:12.4f} {f[2]:11.1f}%")
        else:
            print(f"{arm:10s} {cells:>28s} {'--':>9s} {'--':>12s}  (needs 2 points)")

    # Adjacent-point marginals say whether the arm is actually linear, which a single fitted slope
    # hides. `relion-end-to-end` §10's gather is issue-bound and flat in bytes; nothing here is
    # guaranteed flat in particle count, and the GPU's was not.
    print("\nadjacent marginals, ms/particle")
    for arm in sorted(arms):
        pts = arms[arm]
        ns = sorted(pts)
        seg = [f"{ns[i]//1000}k-{ns[i+1]//1000}k:{(pts[ns[i+1]]-pts[ns[i]])/(ns[i+1]-ns[i])*1e3:.3f}"
               for i in range(len(ns) - 1)]
        print(f"  {arm:10s} {' '.join(seg)}")
    return fits


if __name__ == "__main__":
    main()
