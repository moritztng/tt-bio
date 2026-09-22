#!/usr/bin/env python3
"""What the OpenFold3 ranking rule actually reduces to on a monomer, from the folds
`of3t-confhead` committed.

Reads `perf/of3t_confhead/fold/*.json` at `origin/wk/of3t-confhead` -- nine seeds x five
samples per arm, 1UBQ, production CLI, per-sample `ptm/iptm/disorder/has_clash/rank_score`
alongside the CA-RMSD to the ground truth. No card, no refold: every number here was already
measured and pushed, this only asks what the four terms were worth.

The rule is `0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash`. The question is how many of
those four terms carry any information on a single-chain target, and what the head would have
served had the weight sat on a term that does.
"""
import json
import subprocess
import sys

import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(np.asarray(a, float)))
    rb = np.argsort(np.argsort(np.asarray(b, float)))
    ra, rb = ra - ra.mean(), rb - rb.mean()
    d = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    return float((ra * rb).sum() / d) if d > 0 else float("nan")


def load(rev, arm, seeds):
    out = []
    for s in seeds:
        path = f"perf/of3t_confhead/fold/{arm}_s{s}.json"
        try:
            blob = subprocess.run(["git", "show", f"{rev}:{path}"], check=True,
                                  capture_output=True, text=True).stdout
        except subprocess.CalledProcessError:
            continue
        d = json.loads(blob)
        rows = [r for r in d["per_sample"] if r.get("kind") == "sample"]
        if rows:
            out.append({"seed": s, "arm": arm, "rows": rows,
                        "aiclk": d.get("aiclk_during"), "wall_s": d.get("wall_s")})
    return out


def analyse(folds, key_true="rmsd_ca"):
    served_shipped, served_rand, served_best, served_worst = [], [], [], []
    served_plddt, picked_rank, spear_rule, spear_plddt, spear_ptm = [], [], [], [], []
    nonzero = {"iptm": 0, "disorder": 0, "has_clash": 0}
    n_samples = 0
    for f in folds:
        r = f["rows"]
        n_samples += len(r)
        for k in nonzero:
            nonzero[k] += sum(1 for x in r if float(x.get(k, 0.0)) != 0.0)
        true = [x[key_true] for x in r]
        rule = [x["rank_score"] for x in r]
        plddt = [x["plddt"] for x in r]
        ptm = [x["ptm"] for x in r]
        pick = int(np.argmax(rule))
        served_shipped.append(true[pick])
        served_plddt.append(true[int(np.argmax(plddt))])
        served_rand.append(float(np.mean(true)))
        served_best.append(min(true))
        served_worst.append(max(true))
        # 0 = the head served the best of the five, 4 = it served the worst.
        picked_rank.append(int(np.argsort(np.argsort(true))[pick]))
        # Higher score should mean LOWER rmsd, so a working rule reads negative.
        spear_rule.append(spearman(rule, true))
        spear_plddt.append(spearman(plddt, true))
        spear_ptm.append(spearman(ptm, true))
    m = lambda v: float(np.mean(v))
    return {
        "folds": len(folds), "samples": n_samples,
        "terms_non_zero_over_all_samples": nonzero,
        "served_A": {"shipped_rule": m(served_shipped), "plddt_selector": m(served_plddt),
                     "random": m(served_rand), "perfect": m(served_best),
                     "worst_case": m(served_worst)},
        "picks_best_of_5": sum(1 for r in picked_rank if r == 0),
        "picks_worst_of_5": sum(1 for r in picked_rank if r == 4),
        "mean_true_rank_of_served": m(picked_rank),
        "spearman_vs_rmsd": {"rank_score": m(spear_rule), "plddt": m(spear_plddt),
                             "ptm": m(spear_ptm),
                             "note": "negative is a working selector; 5 samples per fold"},
    }


if __name__ == "__main__":
    rev = sys.argv[1] if len(sys.argv) > 1 else "origin/wk/of3t-confhead"
    out = {"source": rev, "target": "1UBQ, 76 CA, production CLI, 5 samples, 200 steps",
           "rule": "0.8*ipTM + 0.2*pTM + 0.5*disorder - 100*has_clash", "arms": {}}
    for arm in ("ship", "fix"):
        folds = load(rev, arm, range(1, 10))
        out["arms"][arm] = analyse(folds)
        out["arms"][arm]["aiclk_during"] = [f["aiclk"] for f in folds]
    print(json.dumps(out, indent=1))
