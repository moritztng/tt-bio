"""Top-1 success of the TFG reference under the report's three policies, with bootstrap CIs over targets.

Policies (report section 2.1): Unguided = unconstrained candidates, top ranking score. Re-rank = the same unguided
candidates, most requested constraints satisfied, ties by ranking score. Guided = the guided candidates, same rule.
Success = interface-mean DockQ >= 0.23; medium = DockQ >= 0.49.

usage: summarize.py SCORES_JSONL TARGETS_FILE [--seeds 101,102,103,104,105] [--boot 1000]
"""
import argparse
import json
from collections import defaultdict

import numpy as np


def top1(cands, key):
    if not cands:
        return None
    if key is None:
        return max(cands, key=lambda r: r["ranking_score"])
    return max(cands, key=lambda r: (r[key][0], r["ranking_score"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scores")
    ap.add_argument("targets")
    ap.add_argument("--seeds", default="101,102,103,104,105")
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    seeds = {int(s) for s in a.seeds.split(",")}
    tids = open(a.targets).read().split()
    by = defaultdict(list)
    for line in open(a.scores):
        r = json.loads(line)
        if r["seed"] in seeds and r["target"] in tids:
            by[(r["target"], r["cond"])].append(r)

    policies = {"unguided": ("unconstrained", None), "rerank_contact": ("unconstrained", "contact"),
                "guided_contact": ("contact", "contact"), "rerank_pocket": ("unconstrained", "pocket"),
                "guided_pocket": ("pocket", "pocket")}
    picked = {p: [] for p in policies}
    complete = [t for t in tids if all(len(by[(t, c)]) == 5 * len(seeds) for c in ("unconstrained", "contact", "pocket"))]
    for t in complete:
        for p, (cond, key) in policies.items():
            picked[p].append(top1(by[(t, cond)], key)["dockq"])
    rng = np.random.default_rng(0)
    n = len(complete)
    idx = rng.integers(0, n, size=(a.boot, n)) if n else None
    out = {"n_targets": n, "n_candidates": 5 * len(seeds), "incomplete": [t for t in tids if t not in complete]}
    for p, v in picked.items():
        v = np.array(v)
        if not n:
            continue
        stat = {"success": (v >= 0.23), "medium": (v >= 0.49), "mean_dockq": v}
        out[p] = {}
        for k, x in stat.items():
            x = x.astype(float)
            b = x[idx].mean(axis=1)
            out[p][k] = [round(float(x.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    for c in ("contact", "pocket"):
        rows = [r for t in complete for r in by[(t, c)]]
        if rows:
            out[f"samples_satisfied_{c}"] = round(float(np.mean([r[c][0] >= r[c][2] for r in rows])), 3)
        rows = [r for t in complete for r in by[(t, "unconstrained")]]
        if rows:
            out[f"samples_satisfied_{c}_unguided"] = round(float(np.mean([r[c][0] >= r[c][2] for r in rows])), 3)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
