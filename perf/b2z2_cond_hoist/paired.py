#!/usr/bin/env python3
"""Read a bracketed timing run: every non-base arm against the base folds that surround it.

Why brackets. A shared box drifts within a rep -- on qb2 the later fold in a rep runs slower than
the earlier one, systematically. Pooling every base fold and dividing medians lets that drift into
the ratio, and the harness's own "A/A floor" (median base pos0 / median base pos-last) MEASURES the
drift rather than the noise the estimator is exposed to, so it is the wrong null. A base,X,base
bracket cancels a linear drift to first order; the reps are then independent of each other, so the
rep-level paired t interval is the honest reading. (`roof-difftx-arith-efficiency` lesson 1: the two
ratios inside an ABBA quad share a rep and are NOT two samples.)

Why the control has to live in the same rep. Pass 1 of `b2z2-cond-hoist-ship` ran the lever and its
A/A control as two separate benchlocked runs. The lever got a quiet box (acquired at loadavg 0.98,
3.6-9.4 % spread) and the control did not (loadavg 2.93, drifted 17 -> 22.6 s), so the control came
back twenty times wider than the thing it was supposed to price and settled nothing. An arm list
like `base,hoist,base,default,base` fixes that by construction: `default` is the shipped default,
which is the base path, so its bracket is an A/A through the IDENTICAL estimator, measured between
the same two folds of the same rep as the lever's. Whatever the box was doing, it was doing it to
both.

    paired.py <timing json> [--arms hoist,default]

Prints each arm's per-rep bracket, its paired saving and 95 % interval. An arm whose interval
excludes zero is resolved; an A/A arm whose interval INCLUDES zero and is narrower than the lever's
is what licenses reading the lever's.
"""
import argparse, json, statistics as st
from pathlib import Path

# two-sided 95 % t quantile at df = n-1, keyed by the SAMPLE SIZE n so the lookup cannot be off by
# one: T[6] is t(0.975, df=5) = 2.571.
T = {3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
     11: 2.228, 12: 2.201, 13: 2.179, 15: 2.145, 20: 2.093, 21: 2.086}

ap = argparse.ArgumentParser()
ap.add_argument("json", type=Path)
ap.add_argument("--arms", default=None, help="comma list; default every non-base arm present")
ap.add_argument("--max-load", type=float, default=None,
                help="keep only brackets where every fold in them had load0 and load1 below "
                     "this. qb2 is shared; a bracket straddling a co-tenant is not a reading.")
a = ap.parse_args()

d = json.loads(a.json.read_text())
warm = [r for r in d["runs"] if not r.get("cold")]
if a.max_load is not None:
    have = [r for r in warm if "load0" in r]
    if not have:
        raise SystemExit("--max-load needs a run recorded by a harness that logs load0/load1")
    def quiet(r):
        return r["load0"] <= a.max_load and r["load1"] <= a.max_load
else:
    def quiet(r):
        return True
arms = a.arms.split(",") if a.arms else sorted({r["arm"] for r in warm} - {"base"})
reps = sorted({r["rep"] for r in warm})

print(f"{a.json}\n  arms in file: {sorted({r['arm'] for r in warm})}  reps: {len(reps)}")
out = {}
for arm in arms:
    rows, diffs = [], []
    for rep in reps:
        rr = sorted((r for r in warm if r["rep"] == rep), key=lambda r: r["pos"])
        for i, r in enumerate(rr):
            if r["arm"] != arm:
                continue
            # nearest base BEFORE and AFTER this fold, inside the same rep
            before = [x for x in rr[:i] if x["arm"] == "base"]
            after = [x for x in rr[i + 1:] if x["arm"] == "base"]
            if not before or not after:
                continue
            if not (quiet(r) and quiet(before[-1]) and quiet(after[0])):
                continue
            b = (before[-1]["fold_s"] + after[0]["fold_s"]) / 2
            diffs.append(b - r["fold_s"])
            rows.append((rep, before[-1]["fold_s"], r["fold_s"], after[0]["fold_s"], b,
                         b - r["fold_s"], b / r["fold_s"]))
    if len(diffs) < 3:
        print(f"\n{arm}: only {len(diffs)} bracket(s), not enough for an interval")
        continue
    n = len(diffs)
    m, sd = st.mean(diffs), st.stdev(diffs)
    se = sd / n ** 0.5
    t = T[n]
    lo, hi = m - t * se, m + t * se
    ratio = st.mean([r[4] for r in rows]) / st.mean([r[2] for r in rows])
    print(f"\n=== {arm} ===")
    print(f"{'rep':>3} {'base<':>8} {arm[:8]:>8} {'base>':>8} {'bracket':>8} {'saved_s':>8} {'ratio':>8}")
    for r in rows:
        print(f"{r[0]:>3} {r[1]:8.3f} {r[2]:8.3f} {r[3]:8.3f} {r[4]:8.3f} {r[5]:8.3f} {r[6]:8.5f}")
    print(f"n={n} brackets, {sum(1 for x in diffs if x > 0)} of {n} positive")
    print(f"paired saving {m:+.4f} s  95 % CI [{lo:+.4f}, {hi:+.4f}]  "
          f"({'EXCLUDES' if lo > 0 or hi < 0 else 'includes'} zero)")
    print(f"fold ratio    {ratio:.5f}x   CI half-width {t * se:.4f} s")
    out[arm] = (m, lo, hi, t * se)

if len(out) > 1 and "default" in out:
    aa = out["default"]
    print("\n--- verdict ---")
    print(f"A/A (default) {aa[0]:+.4f} s, half-width {aa[3]:.4f} s, "
          f"{'includes' if aa[1] <= 0 <= aa[2] else 'EXCLUDES'} zero")
    for arm, v in out.items():
        if arm == "default":
            continue
        ok = (v[1] > 0 or v[2] < 0) and (aa[1] <= 0 <= aa[2]) and abs(v[0]) > aa[3]
        print(f"{arm}: {v[0]:+.4f} s [{v[1]:+.4f}, {v[2]:+.4f}] -> "
              f"{'RESOLVED, clears its own A/A' if ok else 'not resolved against this A/A'}")
