#!/usr/bin/env python3
"""Quick Phase 5 signal on the partial ranker_scores.csv.

Computes per generator:
  - top-1 DockQ (best by each ranker's score), oracle DockQ (best actual), random DockQ (mean)
  - fraction of oracle gap recovered = (ranked_top1 - random) / (oracle - random)
  - per-target Spearman (median across targets) vs global Spearman
    (the 0.80-vs-0.28 pathology diagnostic from D9)

This is a PARTIAL signal while Tier A is still generating. Coverage is computed
from the CSV and reported at the end -- never hardcoded, because a stale
"95 of 328" printed under a run that actually covered something else is exactly
how a preliminary number gets quoted as a final one.
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
# Generator coverage is read from the CSV, never assumed. The previous version hardcoded both
# this line and the analysis loop to protenix-v2 and boltz2 while announcing "opendde: 0 (not yet
# run)" -- so once opendde-abag started completing, 11 folds / 550 rows of the strongest generator
# by DockQ were dropped from every number below while the header insisted they did not exist.
GEN_ORDER = ["protenix-v2", "opendde-abag", "boltz2"]
_counts = {g: sum(1 for (gg, _) in folds if gg == g) for g in GEN_ORDER}
for _g in sorted({g for g, _ in folds} - set(GEN_ORDER)):   # anything new shows up rather than vanishing
    GEN_ORDER.append(_g)
    _counts[_g] = sum(1 for (gg, _) in folds if gg == _g)
print("  " + ", ".join("%s: %d targets" % (g, _counts[g]) for g in GEN_ORDER))
_learned_hdr = [c for c in ("deeprank_ab", "abag_rank")
                if any(r.get(c) not in (None, "") for v in folds.values() for r in v)]
print("  learned rankers (DeepRank-Ab, ABAG-Rank): %s"
      % (", ".join(_learned_hdr) if _learned_hdr else "NOT run yet"))
print("=" * 80)

for gen in GEN_ORDER:
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
_planned = 164 * 3
_have = len(folds)
_gens_missing = sorted({"protenix-v2", "boltz2", "opendde-abag"} - {g for g, _ in folds})
_learned = [c for c in ("deeprank_ab", "abag_rank")
            if any(r.get(c) not in (None, "") for v in folds.values() for r in v)]
print("NOTE: partial data — %d of %d planned (target,generator) pairs (%.0f%%)."
      % (_have, _planned, 100.0 * _have / _planned))
if _gens_missing:
    print("      generators with no data yet: %s" % ", ".join(_gens_missing))
print("      learned rankers present: %s" % (", ".join(_learned) if _learned else "none"))
print("Numbers are preliminary. The per-target-vs-global Spearman gap (the 0.80-vs-0.28")
print("pathology from btag136) is the headline diagnostic — watch whether global Spearman")
print("is high while per-target median is near zero.")
