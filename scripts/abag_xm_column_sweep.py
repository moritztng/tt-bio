#!/usr/bin/env python3
"""Per-generator fill rate for every column of the ranker-score table.

Three times on this campaign a per-fold component passed on one generator and was broken on the
others (paired-MSA, DeepRank-Ab chain ids, ABAG-Rank's interpreter), and each time the artifact
looked complete because the hole was one column on a subset of folds. So check every column against
every generator rather than only the two that were already caught.

Reads the existing CSV, so it costs nothing and reflects what the pipeline actually produced.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1
            else "/home/ttuser/abag_xm/tier_a/ranker_scores.csv")
rows = list(csv.DictReader(open(path)))
if not rows:
    sys.exit(f"{path}: no rows")

cols = [c for c in rows[0] if c not in ("target", "gen", "rank")]
gens = sorted({r["gen"] for r in rows})
per_gen_rows = defaultdict(int)
filled = defaultdict(lambda: defaultdict(int))
targets = defaultdict(set)
for r in rows:
    g = r["gen"]
    per_gen_rows[g] += 1
    targets[g].add(r["target"])
    for c in cols:
        if r[c] not in ("", None):
            filled[c][g] += 1

print(f"{path}\n{len(rows)} rows, {len(gens)} generators\n")
print(f"{'column':18}" + "".join(f"{g:>16}" for g in gens))
print("-" * (18 + 16 * len(gens)))
suspects = []
for c in cols:
    cells = []
    for g in gens:
        n, tot = filled[c][g], per_gen_rows[g]
        cells.append(f"{n}/{tot}")
    print(f"{c:18}" + "".join(f"{x:>16}" for x in cells))
    # A column that is full for one generator and empty for another is the shape that has bitten
    # three times. Fully-empty-everywhere is a different thing (not requested) and not flagged.
    fracs = {g: (filled[c][g] / per_gen_rows[g] if per_gen_rows[g] else 0) for g in gens}
    if max(fracs.values()) > 0.5 and min(fracs.values()) < 0.5:
        suspects.append((c, fracs))

print(f"\nrows per generator: {dict(per_gen_rows)}")
print(f"targets per generator: { {g: len(t) for g, t in targets.items()} }")

print()
if suspects:
    print("!! ASYMMETRIC COLUMNS (full for one generator, empty for another):")
    for c, fr in suspects:
        print(f"   {c}: " + ", ".join(f"{g}={v:.0%}" for g, v in fr.items()))
    sys.exit(1)
empty_all = [c for c in cols if all(filled[c][g] == 0 for g in gens)]
print("no asymmetric columns: every column is either populated for all generators or for none")
if empty_all:
    print(f"empty for ALL generators (expected if not requested in this run): {empty_all}")
