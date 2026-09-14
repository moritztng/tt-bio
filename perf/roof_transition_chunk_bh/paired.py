#!/usr/bin/env python3
"""Paired analysis of the fold A/B: each test fold against the two shipped folds bracketing it.

Median against median throws the pairing away, which is the whole reason the shipped arm runs on
both sides of the test arm. Paired, the box's slow drift cancels inside a rep and what is left is
the arm. The A/A floor is computed the same way, shipB against shipA, so the two are comparable
and the floor can actually veto the delta rather than sitting beside it as a caveat.

    paired.py out/foldab_c0.json [more.json ...]
"""
import json
import statistics as st
import sys


def reps_of(folds, size):
    """Group into [{shipA, test, shipB}], starting a new rep at each shipA."""
    out = []
    for f in folds:
        if f["size"] != size:
            continue
        if f["slot"] == "shipA":
            out.append({})
        if out:
            out[-1][f["slot"]] = f
    return [r for r in out if {"shipA", "test", "shipB"} <= set(r)]


def se(x):
    return st.stdev(x) / len(x) ** 0.5 if len(x) > 1 else float("nan")


def main():
    folds = []
    for p in sys.argv[1:]:
        folds += [f for f in json.load(open(p))["folds"] if "error" not in f]
    for size in sorted({f["size"] for f in folds}):
        reps = reps_of(folds, size)
        n = len(reps)
        ship = st.median([r[s]["wall_s"] for r in reps for s in ("shipA", "shipB")])
        d_arm = [r["test"]["wall_s"] - (r["shipA"]["wall_s"] + r["shipB"]["wall_s"]) / 2
                 for r in reps]
        d_aa = [r["shipB"]["wall_s"] - r["shipA"]["wall_s"] for r in reps]
        m, s = st.mean(d_arm), se(d_arm)
        loads = [f["loadavg"] for f in folds if f["size"] == size]
        print(f"size {size}  reps={n}  arm={reps[0]['test']['arm']}  "
              f"ship median {ship:.3f} s  loadavg {min(loads):.1f}-{max(loads):.1f}")
        print(f"  arm   {m:+.3f} s  se {s:.3f}  ({100 * m / ship:+.2f} %)  "
              f"faster in {sum(1 for d in d_arm if d < 0)}/{n} reps")
        print(f"  A/A   {st.mean(d_aa):+.3f} s  se {se(d_aa):.3f}  "
              f"per-fold sd {st.stdev(d_aa):.3f} s ({100 * st.stdev(d_aa) / ship:.2f} %)")
        print(f"  95 % CI on the arm [{m - 1.96 * s:+.3f}, {m + 1.96 * s:+.3f}] s  "
              f"ratio {ship / (ship + m):.4f}x  "
              f"{'RESOLVED' if m + 1.96 * s < 0 else 'NOT RESOLVED (CI spans zero)'}")
        print(f"  digests {sorted({r[k]['digest'] for r in reps for k in r})}")


if __name__ == "__main__":
    main()
