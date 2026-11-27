"""r4t-scale: grade trunk_step DP runs into one scaling-curve row each.

  python3 perf/r4t_scale/trunk_curve.py DIR1 DIR2 ...   (each DIR holds rank<r>.log, or A.log for a 1-chip run)

Per run: step time = the slowest rank per step (a DP step ends when the last rank has the reduced gradient),
median over steps 3.., samples/s = world / that, digests_agree = every step's master digest equal on every
rank, AICLK min/median over every rank's in-step samples. eff is against the first DIR (the 1-chip point).
"""
import json
import statistics as st
import sys
from pathlib import Path


def rows(d):
    logs = sorted(Path(d).glob("rank*.log")) or [Path(d, "A.log")]
    return [[json.loads(ln) for ln in open(f) if ln.startswith('{"step"')] for f in logs]


def grade(d):
    rk = rows(d)
    k = min(len(r) for r in rk)
    warm = range(2, k)
    step = st.median(max(r[i]["step_s"] for r in rk) for i in warm)
    world = len(rk)
    return dict(dir=str(d), world=world, steps=k, median_step_s=round(step, 3), samples_per_s=round(world / step, 4),
                split={x: round(st.median(r[i][x] for r in rk for i in warm), 3)
                       for x in ("fwd_s", "bwd_s", "fold_bwd_s", "opt_s", "rebuild_s")},
                digests_agree=all(len({r[i]["master_digest"] for r in rk}) == 1 for i in range(k)),
                digests=[rk[0][i]["master_digest"] for i in range(k)],
                aiclk_min=min(r[i]["aiclk"][0] for r in rk for i in warm),
                aiclk_med=st.median(r[i]["aiclk_med"] for r in rk for i in warm))


if __name__ == "__main__":
    gs = [grade(d) for d in sys.argv[1:]]
    for g in gs:
        g["eff"] = round(g["samples_per_s"] / (g["world"] * gs[0]["samples_per_s"] / gs[0]["world"]), 3)
        print(json.dumps(g))
