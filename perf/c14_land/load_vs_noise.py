#!/usr/bin/env python3
"""Does host load set the fold-to-fold noise, or only the fold time?

ask 9226 asks Moritz to pause a 5-day training job so C14 can measure. The case for it has so far
been a LEVEL shift, +5.66 s per fold, which a paired A/B cancels and which therefore is not on its
own a reason to pause anything. The question that decides whether pausing buys a measurement is
different: does the neighbour also widen the fold-to-fold SPREAD? Spread does not cancel in a
paired design, it is what the session's floor is made of, and it is the whole reason a +0.047 s
lever is undecidable on this box.

This reads the 180 folds of the completed APB session, which carry a per-fold loadavg1 sample, and
regresses |fold - leg median| on that load. Slope > 0 means a quiet host buys resolution and not
just smaller numbers.
"""
import json
import statistics
import sys

d = json.loads(open(sys.argv[1]).read())
pts = []
for b in d["blocks"]:
    r = b.get("result")
    if not r:
        continue
    med = statistics.median(f["fold_s"] for f in r["folds"])
    for f in r["folds"]:
        if f.get("loadavg1") is not None:
            pts.append((f["loadavg1"], abs(f["fold_s"] - med), f["fold_s"]))

lo = [p for p in pts if p[0] < 6.5]
hi = [p for p in pts if p[0] >= 6.5]


def rms(xs):
    return (sum(x * x for x in xs) / len(xs)) ** 0.5


n = len(pts)
xs = [p[0] for p in pts]
ys = [p[1] for p in pts]
mx, my = statistics.mean(xs), statistics.mean(ys)
sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
sxx = sum((x - mx) ** 2 for x in xs)
slope = sxy / sxx
r = sxy / (sxx * sum((y - my) ** 2 for y in ys)) ** 0.5

print("folds with a load sample   %d" % n)
print("loadavg1 range             %.2f - %.2f, mean %.2f" % (min(xs), max(xs), mx))
print()
print("abs deviation from leg median vs loadavg1:  slope %+.4f s per unit load, r = %+.3f"
      % (slope, r))
print("  load < 6.5  n=%3d   rms deviation %.4f s   mean fold %.3f s"
      % (len(lo), rms([p[1] for p in lo]), statistics.mean(p[2] for p in lo)))
print("  load >= 6.5 n=%3d   rms deviation %.4f s   mean fold %.3f s"
      % (len(hi), rms([p[1] for p in hi]), statistics.mean(p[2] for p in hi)))
print()
sd_lo, sd_hi = rms([p[1] for p in lo]) * 1.253, rms([p[1] for p in hi]) * 1.253
print("implied per-fold sd        %.4f s quiet-ish vs %.4f s loud  (%.2fx)"
      % (sd_lo, sd_hi, sd_hi / sd_lo))
print("what that does to a 12 fold x 12 block session's floor:")
for tag, sd in (("observed mix", 0.3291), ("at load < 6.5", sd_lo)):
    sd_m = 1.2 * sd / 12 ** 0.5
    print("  %-14s resolves %.4f s" % (tag, 2 * (sd_m ** 2 * 1.5) ** 0.5 / 12 ** 0.5))
