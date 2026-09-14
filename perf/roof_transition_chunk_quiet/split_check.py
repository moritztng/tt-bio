#!/usr/bin/env python3
"""Did the cotenant that shared the host for the first three reps change the answer?

A cotenant left node 1 partway through this session, so the box got quieter mid-run. The paired
design already absorbs a step change in host load -- both arms of a rep sit either side of it --
but "absorbs it" is a claim, so this splits the reps at the departure and prints the arm delta on
each side. If the two halves agree inside their standard errors the load change is not the result.

    split_check.py <foldab.json> <loadavg threshold>
"""
import json
import statistics as st
import sys

sys.path.insert(0, __file__.rsplit("/", 2)[0] + "/roof_transition_chunk_bh")
from paired import reps_of, se  # noqa: E402 -- the same pairing, not a second implementation


def report(name, reps):
    if len(reps) < 2:
        print(f"  {name:<22} n={len(reps)} -- too few to quote")
        return
    d = [r["test"]["wall_s"] - (r["shipA"]["wall_s"] + r["shipB"]["wall_s"]) / 2 for r in reps]
    a = [r["shipB"]["wall_s"] - r["shipA"]["wall_s"] for r in reps]
    m, s = st.mean(d), se(d)
    print(f"  {name:<22} n={len(reps):<3} arm {m:+.3f} s  se {s:.3f}  "
          f"CI [{m - 1.96 * s:+.3f}, {m + 1.96 * s:+.3f}]  A/A {st.mean(a):+.3f} s se {se(a):.3f}")


def main():
    d = json.load(open(sys.argv[1]))
    thr = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    reps = reps_of([f for f in d["folds"] if "error" not in f], 512)
    hot = [r for r in reps if max(r[k]["loadavg"] for k in ("shipA", "test", "shipB")) > thr]
    cold = [r for r in reps if r not in hot]
    print(f"512 aa, split at peak-in-rep loadavg {thr}")
    report(f"with cotenant", hot)
    report(f"alone on the box", cold)
    report(f"all reps", reps)


if __name__ == "__main__":
    main()
