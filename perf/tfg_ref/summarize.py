"""Top-1 success of the TFG reference under the report's three policies, with bootstrap CIs over targets.

Policies (report section 2.1): Unguided = unconstrained candidates, top ranking score. Re-rank = the same unguided
candidates, most requested constraints satisfied, ties by ranking score. Guided = the guided candidates, same rule.
Success = interface-mean DockQ >= 0.23; medium = DockQ >= 0.49.

usage: summarize.py SCORES_JSONL [SCORES_JSONL ...] TARGETS_FILE [--seeds 101,...] [--guided-seeds 101,...] [--boot 1000]
--guided-seeds (default: --seeds) sets the candidates of the guided policies, so a partial guided arm can be read
beside a complete unguided one. Each policy reports its own target count and candidates per target.
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
    ap.add_argument("files", nargs="+", help="score JSONL files, then the targets file")
    ap.add_argument("--seeds", default="101,102,103,104,105")
    ap.add_argument("--guided-seeds", default=None)
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    *scores, targets = a.files
    useeds = {int(s) for s in a.seeds.split(",")}
    gseeds = {int(s) for s in (a.guided_seeds or a.seeds).split(",")}
    seeds = {"unconstrained": useeds, "contact": gseeds, "pocket": gseeds}
    tids = open(targets).read().split()
    by = defaultdict(list)
    for f in scores:
        for line in open(f):
            r = json.loads(line)
            if r["target"] in tids and r["seed"] in seeds[r["cond"]]:
                by[(r["target"], r["cond"])].append(r)

    policies = {"unguided": ("unconstrained", None), "rerank_contact": ("unconstrained", "contact"),
                "guided_contact": ("contact", "contact"), "rerank_pocket": ("unconstrained", "pocket"),
                "guided_pocket": ("pocket", "pocket")}
    full = {c: [t for t in tids if len(by[(t, c)]) == 5 * len(seeds[c])] for c in seeds}
    rng = np.random.default_rng(0)
    out = {"targets": len(tids)}
    for p, (cond, key) in policies.items():
        done = full[cond]
        if not done:
            continue
        v = np.array([top1(by[(t, cond)], key)["dockq"] for t in done])
        idx = rng.integers(0, len(done), size=(a.boot, len(done)))
        out[p] = {"n_targets": len(done), "n_candidates": 5 * len(seeds[cond])}
        for k, x in {"success": v >= 0.23, "medium": v >= 0.49, "mean_dockq": v}.items():
            x = x.astype(float)
            b = x[idx].mean(axis=1)
            out[p][k] = [round(float(x.mean()), 3), round(float(np.percentile(b, 2.5)), 3), round(float(np.percentile(b, 97.5)), 3)]
    for c in ("contact", "pocket"):
        rows = [r for t in full[c] for r in by[(t, c)]]
        if rows:
            out[f"samples_satisfied_{c}"] = round(float(np.mean([r[c][0] >= r[c][2] for r in rows])), 3)
        rows = [r for t in full["unconstrained"] for r in by[(t, "unconstrained")]]
        if rows:
            out[f"samples_satisfied_{c}_unguided"] = round(float(np.mean([r[c][0] >= r[c][2] for r in rows])), 3)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
