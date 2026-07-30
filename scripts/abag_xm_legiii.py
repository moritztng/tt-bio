#!/usr/bin/env python3
"""Leg (iii): ranked-vs-oracle success, the harness sanity check against OpenDDE's published numbers.

OpenDDE reports 66.4 (ranked) / 80.1 (oracle) percent on 2026ARK-AB at ASSEMBLY level. Our labels are
per-interface on the declared antibody-antigen chain pair, which is a strictly harder quantity -- the
per-interface-vs-wave-averaged measurement in the dataset card found wave-averaged scoring calls
200/200 models acceptable where per-interface calls 106/200. So the check is that we land in a sensible
region and that ranked sits below oracle by a plausible margin, NOT that we reproduce 66.4/80.1.

Definitions, stated because they are the whole content of the check:
  oracle success  -- at least one of the fold's samples reaches the DockQ threshold
  ranked success  -- the sample the ranker puts FIRST reaches the threshold
  gap             -- oracle minus ranked, i.e. what a perfect ranker would add

Reported per generator and per ranker, because "ranked" is only defined relative to a ranker, and the
native confidence is the one OpenDDE's number corresponds to.

    abag_xm_legiii.py [ranker_scores.csv] [--threshold 0.23]
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("csv_path", nargs="?",
                default=str(Path.home() / "abag_xm" / "tier_a" / "ranker_scores.csv"))
ap.add_argument("--threshold", type=float, default=0.23,
                help="DockQ threshold for success (0.23 = CAPRI acceptable)")
a = ap.parse_args()

path = Path(a.csv_path)
if not path.exists():
    sys.exit(f"no ranker table at {path}")
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit(f"{path} is empty")

RANKERS = ["ranking_score", "iptm", "ptm", "complex_plddt", "pdockq2", "ipsae", "anticonf", "pss"]
folds = defaultdict(list)
for r in rows:
    folds[(r["gen"], r["target"])].append(r)

print(f"source: {path}")
print(f"{len(folds)} (generator, target) folds, {len(rows)} samples, "
      f"threshold DockQ >= {a.threshold}\n")

CAPRI = [("high", 0.80), ("medium", 0.49), ("acceptable", 0.23)]


def band(dq):
    for name, lo in CAPRI:
        if dq >= lo:
            return name
    return "incorrect"


gens = sorted({g for g, _ in folds})
for gen in gens:
    gf = {k: v for k, v in folds.items() if k[0] == gen}
    n = len(gf)
    oracle_hits, best_bands = 0, defaultdict(int)
    for _, rs in gf.items():
        dqs = [float(r["dockq"]) for r in rs if r.get("dockq")]
        if not dqs:
            continue
        if max(dqs) >= a.threshold:
            oracle_hits += 1
        best_bands[band(max(dqs))] += 1
    print(f"### {gen}  ({n} targets)")
    print(f"  oracle (best of {len(next(iter(gf.values())))}): "
          f"{oracle_hits}/{n} = {100 * oracle_hits / n:.1f}%   "
          f"bands { {k: best_bands[k] for k, _ in CAPRI} } incorrect={best_bands['incorrect']}")
    for rk in RANKERS:
        hits, usable = 0, 0
        for _, rs in gf.items():
            cand = [r for r in rs if r.get(rk) and r.get("dockq")]
            if not cand:
                continue
            usable += 1
            top = max(cand, key=lambda r: float(r[rk]))
            if float(top["dockq"]) >= a.threshold:
                hits += 1
        if not usable:
            continue
        gap = 100 * oracle_hits / n - 100 * hits / usable
        star = "  <- native confidence, comparable to OpenDDE's number" \
            if rk == "ranking_score" else ""
        print(f"  ranked by {rk:14} {hits}/{usable} = {100 * hits / usable:5.1f}%   "
              f"gap to oracle {gap:5.1f} pts{star}")
    print()

print("Reference: OpenDDE reports 66.4 ranked / 80.1 oracle on 2026ARK-AB at ASSEMBLY level. Ours is "
      "per-interface on the declared chain pair, a strictly harder quantity, so lower absolute numbers "
      "are expected. The check is the region and the ranked<oracle margin, not the value.")
if len(folds) < 492:
    print(f"\n!! PARTIAL: {len(folds)} of 492 folds. The completed folds are whatever the slices "
          f"reached first, so this carries an ordering bias and is indicative only.")
