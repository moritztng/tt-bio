#!/usr/bin/env python3
"""Quick Phase 5 signal on the partial ranker_scores.csv.

Computes per generator:
  - top-1 DockQ (best by each ranker's score), oracle DockQ (best actual), random DockQ (mean)
  - fraction of oracle gap recovered = (ranked_top1 - random) / (oracle - random)
  - per-target Spearman (median across targets) vs global Spearman
    (the 0.80-vs-0.28 pathology diagnostic from D9)

This is a PARTIAL signal (95 of 328 planned pairs, protenix+boltz2 only, no
opendde, no learned rankers). Numbers are preliminary and will change.
"""
import csv, json, sys, os, math
from collections import defaultdict
from pathlib import Path

try:
    from scipy.stats import spearmanr
except Exception:
    spearmanr = None
    print("WARNING: scipy not available, using numpy spearman fallback")

def spearman(a, b):
    if spearmanr:
        r, _ = spearmanr(a, b)
        return r
    # fallback: rank-based
    import numpy as np
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    n = len(a)
    if n < 2:
        return float("nan")
    ma = ra.mean(); mb = rb.mean()
    da = ra - ma; db = rb - mb
    denom = math.sqrt((da**2).sum() * (db**2).sum())
    return float((da*db).sum() / denom) if denom else float("nan")

CSV = Path.home() / "abag_xm" / "tier_a" / "ranker_scores.csv"
if not CSV.exists():
    print("ranker_scores.csv not found at", CSV)
    sys.exit(1)

# rankers to evaluate (columns that are score-like, higher=better)
RANKERS = ["iptm", "ptm", "ranking_score", "complex_plddt", "pdockq2", "ipsae", "anticonf", "pss"]
# oracle label
LABEL = "dockq"

# group rows by (gen, target)
folds = defaultdict(list)
for r in csv.DictReader(open(CSV)):
    folds[(r["gen"], r["target"])].append(r)

print("=" * 80)
print("PARTIAL Phase 5 signal — %d (gen,target) pairs, %d rows" % (len(folds), sum(len(v) for v in folds.values())))
print("  protenix-v2: %d targets, boltz2: %d targets, opendde: 0 (not yet run)" % (
    sum(1 for (g,t) in folds if g=="protenix-v2"),
    sum(1 for (g,t) in folds if g=="boltz2")))
print("  learned rankers (DeepRank-Ab, ABAG-Rank): NOT run yet")
print("=" * 80)

for gen in ["protenix-v2", "boltz2"]:
    gen_folds = {k: v for k, v in folds.items() if k[0] == gen}
    if not gen_folds:
        continue
    print("\n### %s (%d targets, %d samples each) ###" % (gen, len(gen_folds), len(next(iter(gen_folds.values())))))

    # per-ranker: top-1 dockq, oracle dockq, random dockq, gap recovered
    print("\n  %-16s %8s %8s %8s %8s" % ("ranker", "top1_dq", "oracle", "random", "gap_rec"))
    for ranker in RANKERS:
        top1s, oracles, randoms = [], [], []
        per_target_spearman = []
        for (g, t), rows in gen_folds.items():
            pairs = [(float(r[ranker]), float(r[LABEL])) for r in rows if r[ranker] and r[LABEL]]
            if len(pairs) < 2:
                continue
            scores = [p[0] for p in pairs]
            dockqs = [p[1] for p in pairs]
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            top1 = dockqs[best_idx]
            oracle = max(dockqs)
            random_dq = sum(dockqs) / len(dockqs)
            top1s.append(top1); oracles.append(oracle); randoms.append(random_dq)
            sp = spearman(scores, dockqs)
            if not math.isnan(sp):
                per_target_spearman.append(sp)
        if not top1s:
            continue
        mean_top1 = sum(top1s) / len(top1s)
        mean_oracle = sum(oracles) / len(oracles)
        mean_random = sum(randoms) / len(randoms)
        gap = mean_oracle - mean_random
        gap_rec = (mean_top1 - mean_random) / gap if gap > 1e-9 else float("nan")
        print("  %-16s %8.4f %8.4f %8.4f %8.4f" % (ranker, mean_top1, mean_oracle, mean_random, gap_rec))

    # global spearman (pool all samples across targets)
    print("\n  %-16s %8s" % ("ranker", "global_sp"))
    for ranker in RANKERS:
        scores, dockqs = [], []
        for (g, t), rows in gen_folds.items():
            for r in rows:
                if r[ranker] and r[LABEL]:
                    scores.append(float(r[ranker]))
                    dockqs.append(float(r[LABEL]))
        sp = spearman(scores, dockqs)
        print("  %-16s %8.4f" % (ranker, sp))

    # per-target spearman summary
    print("\n  per-target Spearman (median across targets):")
    for ranker in RANKERS:
        pts = []
        for (g, t), rows in gen_folds.items():
            pairs = [(float(r[ranker]), float(r[LABEL])) for r in rows if r[ranker] and r[LABEL]]
            if len(pairs) >= 3:
                scores = [p[0] for p in pairs]
                dockqs = [p[1] for p in pairs]
                sp = spearman(scores, dockqs)
                if not math.isnan(sp):
                    pts.append(sp)
        if pts:
            pts.sort()
            med = pts[len(pts)//2]
            print("  %-16s median=%.4f  (n=%d, q25=%.3f, q75=%.3f)" % (ranker, med, len(pts), pts[len(pts)//4], pts[3*len(pts)//4]))

print("\n" + "=" * 80)
print("NOTE: partial data — 95 of 328 planned pairs. No opendde, no learned rankers.")
print("Numbers are preliminary. The per-target-vs-global Spearman gap (the 0.80-vs-0.28")
print("pathology from btag136) is the headline diagnostic — watch whether global Spearman")
print("is high while per-target median is near zero.")
