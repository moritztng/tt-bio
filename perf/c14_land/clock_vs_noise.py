#!/usr/bin/env python3
"""Is the fold-to-fold spread governed by the AICLK rather than by host load?

load_vs_noise.py found host loadavg explains none of it over 4.5-10.25. The standing clock
discipline says the AICLK sets the fold time, and every fold in this session carries its own
during-fold clock samples, so the question is directly answerable from data already taken.

Each arm's CIF digest is single-valued across all 180 folds, so every fold computed exactly the
same thing. Whatever moves the seconds is therefore machine state, not the model.
"""
import json
import statistics
import sys

d = json.loads(open(sys.argv[1]).read())
rows = []
for b in d["blocks"]:
    r = b.get("result")
    if not r:
        continue
    med = statistics.median(f["fold_s"] for f in r["folds"])
    for f in r["folds"]:
        c = f["clock"]
        rows.append({"dev": f["fold_s"] - med, "fold": f["fold_s"], "arm": b["arm"],
                     "mean": c["aiclk_mean"], "min": c["aiclk_min"], "pw": c.get("power_w_mean")})


def corr(xs, ys):
    mx, my = statistics.mean(xs), statistics.mean(ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    return sxy / (sxx * syy) ** 0.5, sxy / sxx


for key in ("mean", "min", "pw"):
    xs = [r[key] for r in rows if r[key] is not None]
    ys = [r["dev"] for r in rows if r[key] is not None]
    rr, slope = corr(xs, ys)
    print("fold deviation vs aiclk_%-5s  r = %+.3f   slope %+.5f s per unit   range %.1f-%.1f"
          % (key, rr, slope, min(xs), max(xs)))

print()
xs = sorted(rows, key=lambda r: r["mean"])
k = len(xs) // 4
print("lowest  quartile clock mean %.1f MHz  ->  mean fold %.3f s, sd %.4f"
      % (statistics.mean(r["mean"] for r in xs[:k]),
         statistics.mean(r["fold"] for r in xs[:k]),
         statistics.stdev(r["dev"] for r in xs[:k])))
print("highest quartile clock mean %.1f MHz  ->  mean fold %.3f s, sd %.4f"
      % (statistics.mean(r["mean"] for r in xs[-k:]),
         statistics.mean(r["fold"] for r in xs[-k:]),
         statistics.stdev(r["dev"] for r in xs[-k:])))
print()
print("fraction of deviation variance explained by aiclk_mean: %.1f %%"
      % (100 * corr([r["mean"] for r in rows], [r["dev"] for r in rows])[0] ** 2))
