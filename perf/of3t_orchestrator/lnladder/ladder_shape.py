#!/usr/bin/env python3
"""Is a K ladder a valid falsifier for the LayerNorm-reduction hypothesis?

`of3t-lnreduce`'s brief (pass 379) pre-registered this two-sided test: the ratio ours/upstream
is graded by the real-position count K, or it is flat in K and the accumulation hypothesis is
dead. This script asks whether that falsifier is sound BEFORE the row spends card time on it.

The object is `dW = sum_positions(dy * xhat)` -- a reduction over K terms of both signs, so the
true sum is small against the sum of magnitudes and every error is amplified by the cancellation
factor C = sum|t| / |sum t|, which itself grows like sqrt(K).

No torch and no device: this is arithmetic about accumulators, simulated exactly in float64 with
a round-to-m-significand-bits step after every add. math.fsum is the reference.
"""
import json, math, pathlib, platform, random, statistics

def rnd(v, m):
    """Round v to m significand bits, round-half-even. Exact for m <= 53."""
    if v == 0.0 or not math.isfinite(v):
        return v
    e = math.floor(math.log2(abs(v)))
    s = 2.0 ** (m - 1 - e)
    return round(v * s) / s

def accumulate(terms, m):
    """Sequential accumulation, rounding the running sum to m significand bits after each add."""
    a = 0.0
    for t in terms:
        a = rnd(a + t, m)
    return a

def draw(K, rng):
    """dy * xhat at one channel over K positions: zero-mean, both signs, so the sum cancels."""
    return [rng.gauss(0.0, 1.0) for _ in range(K)]

KS = [64, 256, 1024, 4096, 16384, 65536, 147456]
MS = {"bf16": 8, "tt_fp32_19b": 19, "tt_fp32_22b": 22, "ieee_fp32": 24}
TRIALS = 160
rng = random.Random(20260922)

rows = []
for K in KS:
    per = {k: [] for k in MS}
    cancel = []
    for _ in range(TRIALS):
        t = draw(K, rng)
        true = math.fsum(t)
        cancel.append(math.fsum(abs(x) for x in t) / abs(true))
        for name, m in MS.items():
            per[name].append(abs(accumulate(t, m) - true) / abs(true))
    row = {"K": K, "cancellation_factor_median": statistics.median(cancel)}
    for name in MS:
        row["rel_err_" + name] = statistics.median(per[name])
    # the two readings the row's falsifier is about: a 2-bit-short accumulator against IEEE fp32
    row["ratio_22b_over_ieee_fp32"] = row["rel_err_tt_fp32_22b"] / row["rel_err_ieee_fp32"]
    row["ratio_19b_over_ieee_fp32"] = row["rel_err_tt_fp32_19b"] / row["rel_err_ieee_fp32"]
    row["ratio_bf16_over_ieee_fp32"] = row["rel_err_bf16"] / row["rel_err_ieee_fp32"]
    rows.append(row)

def slope(xs, ys):
    """Least-squares slope of log2(y) against log2(x). Flat in K means ~0."""
    lx = [math.log2(x) for x in xs]; ly = [math.log2(y) for y in ys]
    mx = sum(lx)/len(lx); my = sum(ly)/len(ly)
    return sum((a-mx)*(b-my) for a,b in zip(lx,ly)) / sum((a-mx)**2 for a in lx)

r22 = [r["ratio_22b_over_ieee_fp32"] for r in rows]
r19 = [r["ratio_19b_over_ieee_fp32"] for r in rows]
rbf = [r["ratio_bf16_over_ieee_fp32"] for r in rows]
def spread(v): return max(v) / min(v)

out = {
    "instrument": "perf/of3t_orchestrator/lnladder/ladder_shape.py",
    "host": platform.node(),
    "question": "of3t-lnreduce's brief makes 'flat in K' the falsifier of the reduction "
                "hypothesis. Is that sound?",
    "method": "sequential accumulation of K zero-mean terms, running sum rounded to m "
              "significand bits after every add, median of %d trials, math.fsum reference, "
              "float64 throughout. No torch, no device." % TRIALS,
    "rows": rows,
    "ratio_spread_over_K": {
        "tt_fp32_22b_over_ieee_fp32": spread(r22),
        "tt_fp32_19b_over_ieee_fp32": spread(r19),
        "bf16_over_ieee_fp32": spread(rbf),
    },
    "slope_log2_ratio_vs_log2_K": {
        "tt_fp32_22b_over_ieee_fp32": slope(KS, r22),
        "tt_fp32_19b_over_ieee_fp32": slope(KS, r19),
        "bf16_over_ieee_fp32": slope(KS, rbf),
        "//": "0 means the ratio is flat in K; 0.5 would mean it grows like sqrt(K)",
    },
    "slope_log2_abs_err_vs_log2_K": {
        n: slope(KS, [r["rel_err_" + n] for r in rows]) for n in MS
    },
    "median_ratio": {
        "tt_fp32_22b_over_ieee_fp32": statistics.median(r22),
        "tt_fp32_19b_over_ieee_fp32": statistics.median(r19),
        "bf16_over_ieee_fp32": statistics.median(rbf),
    },
    "predicted_ratio_from_bit_deficit": {
        "tt_fp32_22b_over_ieee_fp32": 2.0 ** 2,
        "tt_fp32_19b_over_ieee_fp32": 2.0 ** 5,
        "bf16_over_ieee_fp32": 2.0 ** 16,
    },
}
p = pathlib.Path(__file__).with_name("LADDER_SHAPE.json")
p.write_text(json.dumps(out, indent=1) + "\n")

print("K        cancel_C   bf16      22bit     24bit(ieee)  r22/ieee  r19/ieee  rbf16/ieee")
for r in rows:
    print("%-8d %-10.1f %-9.2e %-9.2e %-12.2e %-9.3f %-9.3f %.3g" % (
        r["K"], r["cancellation_factor_median"], r["rel_err_bf16"], r["rel_err_tt_fp32_22b"],
        r["rel_err_ieee_fp32"], r["ratio_22b_over_ieee_fp32"], r["ratio_19b_over_ieee_fp32"],
        r["ratio_bf16_over_ieee_fp32"]))
print()
for k, v in out["slope_log2_ratio_vs_log2_K"].items():
    if k != "//":
        print("slope log2(ratio)/log2(K) for %-32s = %+0.4f  (0 = flat in K)" % (k, v))
print()
for k, v in out["slope_log2_abs_err_vs_log2_K"].items():
    print("slope log2(abs rel err)/log2(K) for %-22s = %+0.4f" % (k, v))
print()
for k, v in out["ratio_spread_over_K"].items():
    print("spread of %-32s over K = %.3fx  (median %.3f, bit-deficit predicts %.3f)" % (
        k, v, out["median_ratio"][k], out["predicted_ratio_from_bit_deficit"][k]))
print("\nwrote", p)
