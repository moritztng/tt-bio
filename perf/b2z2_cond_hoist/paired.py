#!/usr/bin/env python3
"""The paired reading of a `base,X,base` bracket run, and why the pooled ratio is not it.

`fold_hoist.py --timing-reps` prints a pooled ratio (median of every base fold over median of
every X fold) and an "A/A floor" of median(base pos0) / median(base pos2). On a run with no
within-rep drift those two are the whole story. This run has drift: base at position 2 is
systematically slower than base at position 0, so the pooled base median is inflated by the
late fold and the pooled ratio is biased UP, while the pos0/pos2 "floor" measures the drift
itself rather than the residual noise the estimator is exposed to.

A `base,X,base` bracket exists to cancel exactly that. Per rep, compare X against the MEAN of
the two base folds that surround it; a linear drift across the rep cancels to first order. The
reps are then independent of each other (different folds, no shared observation), so a
rep-level paired t interval is the honest one -- unlike the two ratios inside an ABBA quad,
which share a rep and are not independent (`roof-difftx-arith-efficiency` lesson 1).

    paired.py <timing json> [--arm hoist]

Prints the per-rep bracket ratios, the paired saving and its 95 % interval.
"""
import argparse, json, statistics as st, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("json", type=Path)
ap.add_argument("--arm", default="hoist")
a = ap.parse_args()

d = json.loads(a.json.read_text())
warm = [r for r in d["runs"] if not r.get("cold")]
reps = sorted({r["rep"] for r in warm})
rows, diffs, ratios = [], [], []
for rep in reps:
    rr = [r for r in warm if r["rep"] == rep]
    base = [r["fold_s"] for r in rr if r["arm"] == "base"]
    arm = [r["fold_s"] for r in rr if r["arm"] == a.arm]
    if len(base) != 2 or len(arm) != 1:
        print(f"rep {rep}: not a bracket ({len(base)} base, {len(arm)} {a.arm}), skipped")
        continue
    b = sum(base) / 2
    diffs.append(b - arm[0])
    ratios.append(b / arm[0])
    rows.append((rep, base[0], arm[0], base[1], b, b - arm[0], b / arm[0]))

print(f"{'rep':>3} {'base0':>8} {a.arm:>8} {'base2':>8} {'bracket':>8} {'saved_s':>8} {'ratio':>8}")
for r in rows:
    print(f"{r[0]:>3} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f} {r[4]:8.3f} {r[5]:8.3f} {r[6]:8.5f}")

n = len(diffs)
assert n >= 2, "need at least two brackets"
m, sd = st.mean(diffs), st.stdev(diffs)
se = sd / n ** 0.5
# two-sided 95 % t quantile at df = n-1, keyed by the SAMPLE SIZE n so the lookup cannot be
# off by one: T[6] is t(0.975, df=5) = 2.571.
T = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306,
     10: 2.262, 11: 2.228, 12: 2.201, 14: 2.145, 19: 2.093}[n]
lo, hi = m - T * se, m + T * se
mean_ratio = st.mean([r[4] for r in rows]) / st.mean([r[2] for r in rows])
print(f"\nn={n} brackets, {sum(1 for x in diffs if x > 0)} of {n} positive")
print(f"paired saving  {m:+.4f} s   95 % CI [{lo:+.4f}, {hi:+.4f}]  "
      f"({'excludes' if lo > 0 or hi < 0 else 'INCLUDES'} zero)")
print(f"fold ratio     {mean_ratio:.5f}x   (mean bracket / mean {a.arm})")
print(f"per-rep ratio  mean {st.mean(ratios):.5f}x  min {min(ratios):.5f}  max {max(ratios):.5f}")
