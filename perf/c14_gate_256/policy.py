#!/usr/bin/env python3
"""Score candidate size-ladder measurement policies on their REPEATABILITY, no baseline needed.

The gate fails a rung pair when |k_check - k_base| > tol, k = log(t2/t1)/log(n2/n1), tol 0.50.
Both sides are draws from the same measurement policy, so the arm is a coin flip exactly when
two independent draws of the same policy disagree by more than tol. That is measurable from the
fold sequences alone, without a recorded cell, which matters here: these folds ran on qb1 p150a
and the cells in question are p300c, and a cross-card cell may not be written from a qb1 number.

Policies, all reading the SAME folds:
  today      the gate as shipped. One fresh process per fold, rep 0 discarded, `reps` draws.
             The discard is its own process, so every KEPT draw is still fold 0 of a fresh one.
  warm1      one warm-up fold inside the measuring process, then `reps` measured folds there.
  warm1x     one warm-up fold inside each of `reps` processes, one measured fold in each.
             Costs `reps` processes instead of one, and is the only warm policy that sees the
             between-process spread at all.

    python3 policy.py --dir out --cells w256:256,w512:512
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics as st
from pathlib import Path


def load(d: Path, cell: str):
    """[[fold runtimes in order], ...], one list per fresh process."""
    seqs = []
    for p in sorted(d.glob(cell + "p*.json")):
        r = json.loads(p.read_text())
        s = [row["runtime_s"] for row in sorted(r["rows"], key=lambda x: x["order"])]
        if s:
            seqs.append(s)
    return seqs


def one(seqs, policy: str, reps: int, stat, rng: random.Random, block: bool):
    """One cell value. block=True draws CONSECUTIVE processes, which is what a record run does.

    The recell found the slow mode is chosen between records, not between reps, so the reps
    inside one record are correlated and an i.i.d. resample overstates what a median of them
    buys. In this cell the modes came in runs of two to three processes, so block resampling
    is the honest model and i.i.d. is the optimistic bound.
    """
    def procs(n, minlen):
        pool = [i for i, q in enumerate(seqs) if len(q) >= minlen]
        if block:
            i = rng.randrange(max(1, len(pool) - n + 1))
            w = pool[i:i + n]
            while len(w) < n:
                w.append(pool[-1])
            return [seqs[j] for j in w]
        return [seqs[rng.choice(pool)] for _ in range(n)]

    if policy == "today":
        v = [q[0] for q in procs(reps, 1)]
    elif policy == "warm1":
        v = procs(1, reps + 1)[0][1:reps + 1]
    else:                                        # warm1x
        v = [q[1] for q in procs(reps, 2)]
    return stat(v)


def folds_cost(seqs, policy: str, reps: int) -> float:
    """Device seconds of folding this policy spends at this rung, medians of what was seen."""
    cold = st.median([s[0] for s in seqs])
    warm = st.median([x for s in seqs for x in s[1:]])
    if policy == "today":
        return (reps + 1) * cold
    if policy == "warm1":
        return cold + reps * warm
    return reps * (cold + warm)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out"))
    ap.add_argument("--cells", default="w256:256,w512:512")
    ap.add_argument("--tol", type=float, default=0.50)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block", action="store_true",
                    help="draw the reps of one cell from CONSECUTIVE processes")
    ap.add_argument("--proc-s", type=float, default=17.0,
                    help="process start + device open, seconds, added once per process")
    a = ap.parse_args()

    cells = {}
    for tok in a.cells.split(","):
        name, rung = tok.split(":")
        cells[int(rung)] = load(a.dir, name)
    rungs = sorted(cells)
    lo, hi = rungs[0], rungs[-1]
    span = math.log(hi / lo)

    for rung in rungs:
        print("%-5d %d processes, folds: %s" % (rung, len(cells[rung]),
              " | ".join(" ".join("%.1f" % x for x in s) for s in cells[rung])))
    print()

    stats = {"median": st.median, "min": min}
    print("%-7s %-4s %-7s %-9s %-9s %-8s %-14s %s"
          % ("policy", "reps", "stat", "med(%d)" % lo, "med(%d)" % hi, "k med",
             "P(|dk|>%.2f)" % a.tol, "gate s/model"))
    rowset = []
    for policy in ("today", "warm1", "warm1x"):
        for reps in (1, 3, 5):
            for sname, stat in stats.items():
                if reps == 1 and sname == "min":
                    continue
                if policy == "warm1" and any(
                        max(len(q) for q in cells[r]) < reps + 1 for r in rungs):
                    continue
                d, d2 = {}, {}
                r1 = random.Random(a.seed)
                r2 = random.Random(a.seed + 1)
                for r in rungs:
                    d[r] = [one(cells[r], policy, reps, stat, r1, a.block)
                            for _ in range(a.n)]
                    d2[r] = [one(cells[r], policy, reps, stat, r2, a.block)
                             for _ in range(a.n)]
                k = [math.log(b / s) / span for s, b in zip(d[lo], d[hi])]
                k2 = [math.log(b / s) / span for s, b in zip(d2[lo], d2[hi])]
                bad = sum(abs(x - y) > a.tol for x, y in zip(k, k2)) / a.n
                nproc = {"today": reps + 1, "warm1": 1, "warm1x": reps}[policy]
                cost = sum(folds_cost(cells[r], policy, reps) + nproc * a.proc_s
                           for r in rungs)
                rowset.append((100 * bad, cost, policy, reps, sname))
                print("%-7s %-4d %-7s %-9.2f %-9.2f %-8.3f %-14.1f %.0f"
                      % (policy, reps, sname, st.median(d[lo]), st.median(d[hi]),
                         st.median(k), 100 * bad, cost))
    print()
    print("ranked by false-red rate, then by gate cost:")
    for bad, cost, policy, reps, sname in sorted(rowset):
        print("  %-7s reps=%d %-7s  %5.1f%%  %5.0f s" % (policy, reps, sname, bad, cost))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
