#!/usr/bin/env python3
"""Score the Blackhole leg with the PAIRED estimator, and its own permutation floor.

`analyze.py` quotes a ratio of per-arm medians and floors it by resampling the base arm's own
draws. That estimator assumes the draws are exchangeable in time. On this run they are not: the
host loadavg climbed monotonically 2.26 -> 9.25 as the 44-leg parity gate on card 0 ramped, and the
base arm's 15 folds spread 15.80 % almost entirely as DRIFT (reps 0-9 sit at 19.5-21.0 s, reps
10-14 at 21.2-22.9 s). An unpaired floor charges that drift to noise and comes out at 1.072x, which
cannot resolve a 2 % lever and does not address the claim either.

The run was built paired for exactly this: all three arms run adjacent in time inside every rep,
and the order reverses on alternate reps, so a linear drift is common-mode within a rep and the
positional bias cancels across rep parity (base sits at slot 1 then 3, mean slot 2; `all3` sits at
slot 2 throughout -- the same mean slot). The statistic that uses the design is the median over
reps of the per-rep ratio.

Its null is not a resample of one arm's draws. Under "the lever does nothing", the three arm LABELS
inside a rep are exchangeable, so the floor is the permutation distribution of the same statistic
under within-rep label shuffles. That null carries the drift and the slot effect with it, because
the permuted labels are drawn from the same three timestamps.

  paired.py out/*.json
"""
import itertools
import json
import random
import statistics
import sys
from pathlib import Path


def per_rep(runs, arms):
    """{rep: {arm: fold_s}} for the timed reps only."""
    reps = {}
    for r in runs:
        if r.get("warmup") or r.get("negative_control"):
            continue
        reps.setdefault(r["rep"], {})[r["arm"]] = r["fold_s"]
    return {k: v for k, v in sorted(reps.items()) if set(v) >= set(arms)}


def paired_ratio(reps, treat, base="base"):
    return statistics.median([v[base] / v[treat] for v in reps.values()])


def boot_ci(reps, treat, iters=40000, seed=0):
    rng = random.Random(seed)
    rs = [v[base_k] / v[treat] for v in reps.values() for base_k in ("base",)]
    out = []
    for _ in range(iters):
        out.append(statistics.median(rng.choices(rs, k=len(rs))))
    out.sort()
    return out[int(0.025 * iters)], out[int(0.975 * iters)]


def perm_floor(reps, arms, iters=40000, seed=0):
    """95th percentile of the paired statistic when the arm labels are shuffled within each rep."""
    rng = random.Random(seed)
    rows = [[v[a] for a in arms] for v in reps.values()]
    i_base, i_treat = 0, 1
    out = []
    for _ in range(iters):
        rr = []
        for row in rows:
            p = row[:]
            rng.shuffle(p)
            rr.append(p[i_base] / p[i_treat])
        m = statistics.median(rr)
        out.append(max(m, 1 / m))
    out.sort()
    return {"p50": out[iters // 2], "p95": out[int(0.95 * iters)], "p99": out[int(0.99 * iters)]}


def score(path):
    d = json.loads(Path(path).read_text())
    arms = d["arms"]
    reps = per_rep(d["runs"], arms)
    e = d["env"]
    n = len(reps)
    loads = {r["rep"]: r["loadavg"] for r in d["runs"] if "rep" in r}
    print(f"\n=== {Path(path).name}  {e['arch']} grid {e['grid']} card "
          f"{e['tt_visible_devices']} commit {e['commit'][:9]}  n={n} paired reps")
    base = [v["base"] for v in reps.values()]
    print(f"    base median {statistics.median(base):.4f} s  spread "
          f"{100 * (max(base) - min(base)) / statistics.median(base):.2f} %  "
          f"loadavg {d['loadavg_range'][0]}-{d['loadavg_range'][1]}  "
          f"(reps 0-9 median {statistics.median(base[:10]):.3f} s, "
          f"10-14 {statistics.median(base[10:]):.3f} s)")
    floor = perm_floor(reps, ["base", "all3", "qkvg"])
    print(f"    permutation floor of the paired median, within-rep label shuffle: "
          f"p95 {floor['p95']:.5f}x  p99 {floor['p99']:.5f}x")
    res = {"file": Path(path).name, "arch": e["arch"], "n": n, "perm_floor": floor,
           "base_median_s": statistics.median(base), "paired": {}}
    for arm in arms:
        if arm == "base":
            continue
        r = paired_ratio(reps, arm)
        lo, hi = boot_ci(reps, arm)
        wins = sum(1 for v in reps.values() if v["base"] > v[arm])
        verdict = "CLEARS" if r > floor["p95"] else "inside the floor"
        print(f"    {arm:5s} paired {r:.5f}x   95 % CI {lo:.5f}-{hi:.5f}x   "
              f"faster in {wins}/{n} reps   {verdict}")
        res["paired"][arm] = {"ratio": r, "ci": [lo, hi], "wins": wins, "verdict": verdict}
    print("    per-rep base/all3: " + " ".join(
        f"{v['base'] / v['all3']:.4f}" for v in reps.values()))
    print("    loadavg per rep:   " + " ".join(f"{loads[k]:6.2f}" for k in reps))
    return res


if __name__ == "__main__":
    out = [score(p) for p in sys.argv[1:]]
    Path(sys.argv[1]).with_name("paired_scored.json").write_text(json.dumps(out, indent=1))
