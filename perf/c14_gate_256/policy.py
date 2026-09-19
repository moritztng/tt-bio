#!/usr/bin/env python3
"""Score candidate size-ladder measurement policies on their REPEATABILITY, no baseline needed.

The gate fails a rung pair when |k_check - k_base| > tol, k = log(t2/t1)/log(n2/n1), tol 0.50.
Both sides are draws from the same measurement policy, so the arm is a coin flip exactly when
two independent draws of the same policy disagree by more than tol. That is measurable from the
fold sequences alone, without a recorded cell, which matters here: these folds ran on qb1 p150a
and the cells in question are p300c, and a cross-card cell may not be written from a qb1 number.

Policies, both reading the SAME folds:
  today      the gate as shipped. One fresh process per fold, rep 0 discarded, median of `reps`
             draws. The discard is its own process, so every KEPT draw is still fold 0 of a
             fresh process.
  warm1      one warm-up fold inside the measuring process, then `reps` measured folds in that
             same process.
Each scored at reps = 1 and reps = 3.

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


def draws(seqs, policy: str, reps: int, rng: random.Random, n: int):
    """n independent draws of one cell value under `policy` at `reps`."""
    out = []
    pool = [q for q in seqs if len(q) >= reps + 1]
    for _ in range(n):
        if policy == "today":
            v = [rng.choice(seqs)[0] for _ in range(reps)]
        else:
            s = rng.choice(pool)
            v = s[1:reps + 1]
        out.append(st.median(v))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("out"))
    ap.add_argument("--cells", default="w256:256,w512:512")
    ap.add_argument("--tol", type=float, default=0.50)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cells = {}
    for tok in a.cells.split(","):
        name, rung = tok.split(":")
        cells[int(rung)] = load(a.dir, name)
    rungs = sorted(cells)
    lo, hi = rungs[0], rungs[-1]
    span = math.log(hi / lo)

    for rung in rungs:
        seqs = cells[rung]
        print("%-5d %d processes, folds: %s" % (rung, len(seqs),
              " | ".join(" ".join("%.1f" % x for x in s) for s in seqs)))
    print()

    print("%-7s %-5s %-9s %-9s %-9s %s"
          % ("policy", "reps", "med(256)", "med(512)", "k median", "P(|dk| > %.2f)" % a.tol))
    for policy in ("today", "warm1"):
        for reps in (1, 3):
            rng = random.Random(a.seed)
            d = {r: draws(cells[r], policy, reps, rng, a.n) for r in rungs}
            k = [math.log(b / s) / span for s, b in zip(d[lo], d[hi])]
            rng2 = random.Random(a.seed + 1)
            d2 = {r: draws(cells[r], policy, reps, rng2, a.n) for r in rungs}
            k2 = [math.log(b / s) / span for s, b in zip(d2[lo], d2[hi])]
            bad = sum(abs(x - y) > a.tol for x, y in zip(k, k2)) / a.n
            print("%-7s %-5d %-9.2f %-9.2f %-9.3f %.1f%%"
                  % (policy, reps, st.median(d[lo]), st.median(d[hi]),
                     st.median(k), 100 * bad))
    print()
    print("fold seconds per model over both rungs (medians of what was measured):")
    for policy in ("today", "warm1"):
        for reps in (1, 3):
            tot = 0.0
            for rung in rungs:
                seqs = cells[rung]
                warm = st.median([s[0] for s in seqs])
                if policy == "today":
                    tot += (reps + 1) * warm
                else:
                    rest = st.median([x for s in seqs for x in s[1:reps + 1]])
                    tot += warm + reps * rest
            print("  %-7s reps=%d  %6.1f s of folds" % (policy, reps, tot))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
