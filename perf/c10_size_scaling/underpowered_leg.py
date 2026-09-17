"""The 512 -> 768 leg, computed HONESTLY on a session the hardware cut short.

768 aa wedged the chip on its sixth fold, so its 800 MHz cell holds 2 accepted folds against a
pre-registered minimum of 4 and reduce.py correctly refuses the rung. That refusal is the right
default and is left in place. This script exists so the leg can still be quoted with its weakness
stated, and it quotes it two ways: the usual standard error, which leans on an sd estimated from
two folds and is therefore the weaker of the two, and a distribution-free MIN/MAX envelope that
leans on no variance estimate at all. If both put the exponent above the pre-registered artifact
threshold of 1.7, the verdict does not depend on the missing folds.
"""
from __future__ import annotations
import json, math, statistics, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE)]
from sizefit import work_two_clock, leg_exponent, size_independent_floor

THRESHOLD = 1.7
run = Path(sys.argv[1])
cells = {}
for size in (512, 768):
    rows = [r for r in json.loads((run / str(size) / "result.json").read_text())["rows"]
            if r.get("valid") and r["label"] != "cold"]
    cells[size] = {}
    for f in (1350, 800):
        t = sorted(r["elapsed_s"] for r in rows if r["clock_MHz"] == f)
        sd = statistics.stdev(t) if len(t) > 1 else None
        cells[size][f] = {"n": len(t), "folds_s": t, "median_s": statistics.median(t),
                          "min_s": min(t), "max_s": max(t), "sd_s": sd,
                          "se_median_s": (1.2533 * sd / math.sqrt(len(t))) if sd else None}

work, env = {}, {}
for size in (512, 768):
    c = cells[size]
    work[size] = work_two_clock(1350, c[1350]["median_s"], c[1350]["se_median_s"],
                                800, c[800]["median_s"], c[800]["se_median_s"])
    g = abs(work[size]["dW_dt_gain"][0])
    env[size] = {"W_min_Mcycles": g * (c[800]["min_s"] - c[1350]["max_s"]),
                 "W_max_Mcycles": g * (c[800]["max_s"] - c[1350]["min_s"])}

leg = leg_exponent(512, work[512]["W_Mcycles"], work[512]["se_W_Mcycles"],
                   768, work[768]["W_Mcycles"], work[768]["se_W_Mcycles"])
leg.update(size_independent_floor(512, work[512]["W_Mcycles"], 768, work[768]["W_Mcycles"]))
ln = math.log(768 / 512)
envelope = {"p_min": math.log(env[768]["W_min_Mcycles"] / env[512]["W_max_Mcycles"]) / ln,
            "p_max": math.log(env[768]["W_max_Mcycles"] / env[512]["W_min_Mcycles"]) / ln}

out = {"leg": "512->768", "cells": cells, "work": work, "min_max_work_envelope": env,
       "leg_standard_error_method": leg, "leg_min_max_envelope_method": envelope,
       "artifact_threshold": THRESHOLD,
       "sigma_above_threshold": (leg["p_work"] - THRESHOLD) / leg["se_p_work"],
       "envelope_entirely_above_threshold": envelope["p_min"] > THRESHOLD,
       "under_strength_cells": {f"{s}@{f}": cells[s][f]["n"] for s in cells for f in cells[s]
                                if cells[s][f]["n"] < 4},
       "preregistered_minimum_folds_per_cell": 4,
       "verdict": "ARTIFACT" if (leg["p_work"] >= THRESHOLD and envelope["p_min"] > THRESHOLD)
                  else "FINDING SURVIVES" if leg["p_work"] < THRESHOLD else "INCONCLUSIVE"}
(run / "underpowered_leg.json").write_text(json.dumps(out, indent=2) + "\n")

for s in (512, 768):
    for f in (1350, 800):
        c = cells[s][f]
        print(f"  {s} aa @ {f:4d} MHz  n={c['n']}  median {c['median_s']:.4f} s  "
              f"[{c['min_s']:.4f}, {c['max_s']:.4f}]  sd {c['sd_s'] and round(c['sd_s'],4)}")
    print(f"  {s} aa  W = {work[s]['W_Mcycles']:9.1f} +- {work[s]['se_W_Mcycles']:.1f} Mcycles   "
          f"F = {work[s]['F_s']:.4f} +- {work[s]['se_F_s']:.4f} s   "
          f"envelope W [{env[s]['W_min_Mcycles']:.1f}, {env[s]['W_max_Mcycles']:.1f}]")
print(f"\n  work ratio   {leg['work_ratio']:.4f} against a token ratio of 1.5000")
print(f"  p_work       {leg['p_work']:.4f} +- {leg['se_p_work']:.4f}   "
      f"({out['sigma_above_threshold']:.1f} sigma ABOVE the {THRESHOLD} artifact threshold)")
print(f"  envelope     p in [{envelope['p_min']:.4f}, {envelope['p_max']:.4f}] with no variance estimate at all")
print(f"  mixture floor {leg['floor_Mcycles']:.1f} Mcycles ({leg['floor_pct_of_W1']:.1f} % of W512): "
      f"{'a size-independent term is implied' if leg['implies_size_independent_term'] else 'NO size-independent term is implied on this leg'}")
print(f"  under-strength cells: {out['under_strength_cells'] or 'none'} "
      f"(pre-registered minimum {out['preregistered_minimum_folds_per_cell']})")
print(f"\n  VERDICT: {out['verdict']}")
