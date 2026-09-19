#!/usr/bin/env python3
"""Where does this session's noise live: inside a leg, or between legs?

The power question the NULL raises is whether to buy more BLOCKS or more FOLDS PER LEG. Those
cost very differently -- a leg pays a fixed model-construction and warmup cost once and then
folds cheaply -- so the answer decides how a longer session should be shaped.

  var(leg median)  is estimated from the A/A draws: var(aa) = 2 * var(leg median).
  var(fold | leg)  is estimated pooled within legs.
If the leg median's variance is close to var(fold)/n_folds, fold-to-fold noise dominates and
more folds per leg buys power at the cheap end. If it is much larger, there is a leg-level
level shift that only more blocks can average down.
"""
import json
import statistics
import sys

d = json.loads(open(sys.argv[1]).read())
legs = [b for b in d["blocks"] if b.get("returncode") == 0 and b.get("result")]
within = []
per_leg_sd = []
for b in legs:
    f = [x["fold_s"] for x in b["result"]["folds"]]
    sd = statistics.stdev(f)
    per_leg_sd.append(sd)
    within += [x - statistics.mean(f) for x in f]
pooled_fold_sd = (sum(x * x for x in within) / (len(within) - len(legs))) ** 0.5

med = {}
for b in legs:
    med[(b["block"], b["pos"])] = statistics.median(x["fold_s"] for x in b["result"]["folds"])
aas = [med[(k, 0)] - med[(k, 2)] for k in sorted({b["block"] for b in legs})
       if (k, 0) in med and (k, 2) in med]
sd_aa = statistics.stdev(aas)
sd_legmed = sd_aa / 2 ** 0.5

nf = len(legs[0]["result"]["folds"])
sd_med_from_folds = 1.2 * pooled_fold_sd / nf ** 0.5   # median of n ~ 1.2x the mean's SE

print("legs                       %d, %d folds each" % (len(legs), nf))
print("pooled within-leg fold sd  %.4f s" % pooled_fold_sd)
print("median within-leg sd       %.4f s" % statistics.median(per_leg_sd))
print("sd(A/A draw)               %.4f s   -> sd(leg median) %.4f s" % (sd_aa, sd_legmed))
print("sd(leg median) predicted   %.4f s   from fold noise alone at n=%d" % (sd_med_from_folds, nf))
excess = sd_legmed ** 2 - sd_med_from_folds ** 2
print("leg-level excess sd        %.4f s   (%s)" % (
    abs(excess) ** 0.5, "leg-level shift dominates" if excess > 0 else "fold noise explains it"))
print()
for nf2 in (5, 8, 12, 16):
    sd_m = (max(excess, 0) + (1.2 * pooled_fold_sd / nf2 ** 0.5) ** 2) ** 0.5
    sd_delta = (sd_m ** 2 * 1.5) ** 0.5     # delta = mean(two bases) - on
    leg_s = 170 + nf2 * 16.0                # ~170 s fixed per leg, ~16 s per fold
    for n in (12, 25):
        floor = 2 * sd_delta / n ** 0.5
        print("folds/leg %2d  blocks %2d  ->  resolves %.4f s   session %.1f h"
              % (nf2, n, floor, 3 * leg_s * n / 3600))
