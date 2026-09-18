#!/usr/bin/env python3
"""Fit `_MM_IN1_BLOCK_TILES` against every group the lever CHANGES, not just the starved ones.

Each group contributes `calls * fold_us * (1 - rung[pick] / rung[fold_bw])`: the ratio comes from
the ladder (one instrument, both rungs, interleaved) and the base from the fold. A group the rule
leaves at its current blocking contributes exactly zero, which is the point -- the first rule this
code shipped scored well on the starved group and paid for it on a wide-output one nobody had
measured.
"""
import json

G = [
    # label, calls, fold_us, fold_bw, out_block_w, k_tiles, rungs {bw: us}
    ("1,512,768,768 starved", 19200, 24.018, 1, 3, 24,
     {1: 26.166, 2: 19.897, 3: 18.286, 4: 17.351, 6: 16.687, 8: 16.311, 12: 16.430, 24: 18.744}),
    ("1,512,768,768 at bw4", 19400, 17.344, 4, 3, 24,
     {1: 27.517, 2: 20.719, 3: 19.229, 4: 18.510, 6: 17.999, 8: 17.811, 12: 18.675, 24: 20.995}),
    ("1,512,768,1536", 15200, 22.628, 4, 5, 24,
     {1: 33.526, 2: 28.143, 3: 26.491, 4: 25.632, 6: 24.396, 8: 23.815, 12: 23.198, 24: 27.339}),
    ("1,512,768,3072", 4800, 40.101, 4, 9, 24,
     {1: 48.584, 2: 43.653, 3: 43.481, 4: 43.174, 6: 44.418, 8: 45.134, 12: 48.083, 24: 55.018}),
    ("1,512,1536,768", 5200, 23.282, 4, 3, 48,
     {1: 44.637, 2: 31.904, 3: 28.595, 4: 26.583, 6: 25.002, 8: 24.697, 12: 24.862, 16: 24.609,
      24: 26.405, 48: 32.220}),
]


def pick(C, obw, kt, rungs):
    best = 0
    for d in range(1, kt + 1):
        if kt % d == 0 and d * obw <= C and d in rungs:
            best = d
    return best or min(rungs)


print("%4s %9s %9s  %s" % ("C", "net_s", "worst", "picks"))
rows = []
for C in list(range(6, 121, 3)):
    tot, worst, picks = 0.0, 1.0, []
    for label, calls, fold_us, fbw, obw, kt, rungs in G:
        d = pick(C, obw, kt, rungs)
        r = rungs[d] / rungs[fbw]
        tot += calls * fold_us * (1 - r) / 1e6
        worst = max(worst, r)
        picks.append("%d" % d)
    rows.append((C, tot, worst, picks))
    print("%4d %9.4f %9.4f  %s" % (C, tot, worst, ",".join(picks)))
best = max(rows, key=lambda r: r[1])
print("\nargmax C=%d net %+.4f s, worst group %.4fx, picks %s"
      % (best[0], best[1], best[2], ",".join(best[3])))
# the no-regression constraint matters as much as the total
safe = [r for r in rows if r[2] <= 1.02]
bs = max(safe, key=lambda r: r[1])
print("best C with no group worse than 1.02x: C=%d net %+.4f s, worst %.4fx, picks %s"
      % (bs[0], bs[1], bs[2], ",".join(bs[3])))
