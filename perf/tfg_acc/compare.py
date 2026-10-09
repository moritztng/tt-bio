"""TT vs upstream GPU on the TFG panel: top-1 success, CAPRI-medium and mean DockQ per policy, with CIs.

Policies (report 2.1): unguided = unconstrained candidates by ranking score; rerank_<c> = the same candidates by
requested constraints satisfied, ties by ranking score; guided_<c> = guided candidates, same rule. Robustness
conditions (c4, wrong1, ...) are "guided" policies on their own requests.

CIs are the report's nested bootstrap: resample targets, then within each target resample its seeds (5 samples
each) and re-pick top-1 from the resampled candidates. Paired statistics reuse one target resample for both arms,
so TT - GPU and guided - unguided are per-target differences. Only targets complete on every compared arm count.

usage: compare.py --tt TT.jsonl [--gpu GPU.jsonl] --targets FILE [--boot 1000] [--json OUT] [--conds ...]
"""
import argparse
import json
from collections import defaultdict

import numpy as np

SUCCESS, MEDIUM = 0.23, 0.49


def load(path, tids, seeds):
    by = defaultdict(lambda: defaultdict(list))  # (target, cond) -> seed -> rows
    for line in open(path):
        r = json.loads(line)
        if r["target"] in tids and r["seed"] in seeds:
            by[(r["target"], r["cond"])][r["seed"]].append(r)
    return by


def pick(rows, key):
    if key is None:
        return max(rows, key=lambda r: r["ranking_score"])
    return max(rows, key=lambda r: (r[key][0], r["ranking_score"]))


def policies(conds):
    out = {"unguided": ("unconstrained", None)}
    for c in conds:
        if c in ("contact", "pocket"):
            out[f"rerank_{c}"] = ("unconstrained", c)
        out[f"guided_{c}"] = (c, "req" if c not in ("contact", "pocket") else c)
    return out


def top1_matrix(by, tids, seeds, pol, draws):
    """(boot+1, T) top-1 DockQ: row 0 the point estimate, row b+1 the top-1 over target j's seeds draws[b, j]."""
    cond, key = pol
    out = np.zeros((draws.shape[0] + 1, len(tids)))
    for j, t in enumerate(tids):
        per_seed = [by[(t, cond)][s] for s in seeds]
        out[0, j] = pick([r for rs in per_seed for r in rs], key)["dockq"]
        cache = {}
        for b in range(draws.shape[0]):
            k = tuple(sorted(draws[b, j]))
            if k not in cache:
                cache[k] = pick([r for i in k for r in per_seed[i]], key)["dockq"]
            out[b + 1, j] = cache[k]
    return out


METRICS = (("success", lambda x: (x >= SUCCESS).astype(float)), ("medium", lambda x: (x >= MEDIUM).astype(float)),
           ("mean_dockq", lambda x: x))


def stats(m, tidx, base=None):
    """point, lo, hi of success / medium / mean DockQ over targets; with `base`, of the paired difference m - base."""
    res = {}
    for name, f in METRICS:
        v = f(m) - (f(base) if base is not None else 0.0)
        boots = np.array([v[b + 1, tidx[b]].mean() for b in range(len(tidx))])
        res[name] = [round(float(v[0].mean()), 3), round(float(np.percentile(boots, 2.5)), 3),
                     round(float(np.percentile(boots, 97.5)), 3)]
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt", required=True)
    ap.add_argument("--gpu")
    ap.add_argument("--targets", required=True)
    ap.add_argument("--conds", default="contact,pocket")
    ap.add_argument("--seeds", default="101,102,103,104,105")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--json")
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(",")]
    tids = [t for t in open(a.targets).read().split() if t]
    conds = a.conds.split(",")
    pols = policies(conds)
    arms = {"tt": load(a.tt, set(tids), set(seeds))}
    if a.gpu:
        arms["gpu"] = load(a.gpu, set(tids), set(seeds))
    need = {"unconstrained", *conds}
    complete = [t for t in tids if all(len(by[(t, c)].get(s, [])) == 5 for by in arms.values() for c in need for s in seeds)]
    out = {"n_targets": len(complete), "targets": complete, "incomplete": [t for t in tids if t not in complete],
           "candidates": 5 * len(seeds), "boot": a.boot}
    if not complete:
        print(json.dumps(out, indent=1))
        return
    rng = np.random.default_rng(0)
    T = len(complete)
    tidx = rng.integers(0, T, size=(a.boot, T))
    draws = rng.integers(0, len(seeds), size=(a.boot, T, len(seeds)))  # one seed resample shared by every arm
    mats = {arm: {p: top1_matrix(by, complete, seeds, pol, draws) for p, pol in pols.items()} for arm, by in arms.items()}
    for arm, ms in mats.items():
        out[arm] = {p: stats(m, tidx) for p, m in ms.items()}
        out[arm]["gain"] = {p: stats(m, tidx, ms["unguided"]) for p, m in ms.items() if p != "unguided"}
        for c in conds:  # share of guided samples that meet their request
            rows = [r for t in complete for s in seeds for r in arms[arm][(t, c)][s]]
            k = c if c in ("contact", "pocket") else "req"
            out[arm][f"samples_satisfied_{c}"] = round(float(np.mean([r[k][0] >= r[k][2] for r in rows])), 3)
    if "gpu" in mats:
        out["tt_minus_gpu"] = {p: stats(mats["tt"][p], tidx, mats["gpu"][p]) for p in pols}
        out["per_target"] = {t: {p: [round(float(mats[arm][p][0, j]), 3) for arm in ("tt", "gpu")] for p in pols}
                             for j, t in enumerate(complete)}
    else:
        out["per_target"] = {t: {p: round(float(mats["tt"][p][0, j]), 3) for p in pols} for j, t in enumerate(complete)}
    s = json.dumps(out, indent=1)
    if a.json:
        open(a.json, "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
