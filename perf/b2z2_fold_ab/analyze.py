#!/usr/bin/env python3
"""Score the fold A/B runs, with the A/A floor recomputed at the n the ratio is quoted at.

`fold_ab3.py` records a split-half floor: it shuffles the base arm's reps, takes the median of one
half against the median of the other, and reports the 95th percentile. That statistic is the floor
of a median-of-3, not of the median-of-7 the ratio is actually quoted with, so it is ~1.5x too wide
and on Blackhole it swallowed a real 1.034x. The floor below bootstraps TWO independent size-n
resamples from the base arm's own draws, which is the null distribution of the statistic being
reported. Same principle CONTEXT's A/A block states: quote the floor of the statistic you report.

  analyze.py out/*.json
"""
import json
import random
import statistics
import sys
from pathlib import Path


def aa_floor(vals, n, iters=40000, seed=0):
    rng = random.Random(seed)
    r = []
    for _ in range(iters):
        a = statistics.median(rng.choices(vals, k=n))
        b = statistics.median(rng.choices(vals, k=n))
        r.append(max(a / b, b / a))
    r.sort()
    return {"p50": r[len(r) // 2], "p95": r[int(0.95 * len(r))], "n": n}


def score(path):
    d = json.loads(Path(path).read_text())
    s, med = d["series_s"], d["median_fold_s"]
    n = len(s["base"])
    floor = aa_floor(s["base"], n)
    e = d["env"]
    print(f"\n=== {Path(path).name}  {e['arch']} grid {e['grid']} card {e['tt_visible_devices']} "
          f"commit {e['commit'][:9]}  n={n}")
    print(f"    base {med['base']:.4f} s   spread "
          f"{100 * (max(s['base']) - min(s['base'])) / med['base']:.2f} %   "
          f"loadavg {d['loadavg_range'][0]}-{d['loadavg_range'][1]}")
    print(f"    A/A floor at n={n}: p95 {floor['p95']:.5f}x   (split-half, as recorded: "
          f"{d['aa_floor']['p95']:.5f}x)")
    for arm in ("all3", "qkvg"):
        r = med["base"] / med[arm]
        verdict = "CLEARS" if r > floor["p95"] else "inside the floor"
        print(f"    {arm:5s} {med[arm]:.4f} s   {r:.5f}x   {med['base'] - med[arm]:+.4f} s   "
              f"{verdict}")
    print(f"    bit-exact across arms {d['bit_exact_across_arms']}  sha "
          f"{d['sha_per_arm']['base'][0]}   negative control breaks "
          f"{d.get('negative_control_breaks')}")
    print(f"    census flat {d['census_flat_per_arm']}  served/rejected all3 "
          f"{ {k: v for k, v in d['census_per_arm']['all3'].items() if k in ('qkvg', 'qkvgb', 'trimul_gout')} }")
    return {"file": Path(path).name, "arch": e["arch"], "n": n, "aa_p95": floor["p95"],
            "ratio": {a: med["base"] / med[a] for a in ("all3", "qkvg")},
            "saved_s": {a: med["base"] - med[a] for a in ("all3", "qkvg")},
            "base_s": med["base"]}


if __name__ == "__main__":
    out = [score(p) for p in sys.argv[1:]]
    Path(__file__).with_name("out").joinpath("scored.json").write_text(json.dumps(out, indent=1))
