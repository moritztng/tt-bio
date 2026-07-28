"""Rank the strict Ab-Ag PSBench candidates by dockq_wave spread across their 200 AF3 models.

A rank correlation is meaningless on a target whose models are all equally good, so leg (i)
picks the targets with the widest interquartile range. Writes dockq_wave_spread.json.
"""
import csv
import json
import pathlib
import statistics

BASE = pathlib.Path("/home/ttuser/abag_xm/psbench")
qs = BASE / "Quality_Scores"
cand = [l.strip() for l in open(BASE / "abag_candidates_strict.txt") if l.strip()]

rows = []
for pid in cand:
    f = qs / f"{pid}_quality_scores.csv"
    if not f.exists():
        continue
    vals = []
    for r in csv.DictReader(open(f)):
        try:
            vals.append(float(r["dockq_wave"]))
        except (TypeError, ValueError):
            pass
    if len(vals) < 100:
        continue
    q1, _, q3 = statistics.quantiles(vals, n=4)
    rows.append({"id": pid, "n": len(vals), "min": round(min(vals), 3),
                 "median": round(statistics.median(vals), 3), "max": round(max(vals), 3),
                 "iqr": round(q3 - q1, 3)})

rows.sort(key=lambda r: -r["iqr"])
json.dump(rows, open(BASE / "dockq_wave_spread.json", "w"), indent=1)

print("candidates with >=100 scored models:", len(rows))
print("%-6s %4s %6s %6s %6s %6s" % ("id", "n", "iqr", "min", "med", "max"))
for r in rows[:25]:
    print("%-6s %4d %6.3f %6.3f %6.3f %6.3f" % (r["id"], r["n"], r["iqr"], r["min"], r["median"], r["max"]))
iqrs = [r["iqr"] for r in rows]
print("iqr: max %.3f  median %.3f  min %.3f" % (max(iqrs), statistics.median(iqrs), min(iqrs)))
print("candidates with iqr >= 0.15:", sum(1 for v in iqrs if v >= 0.15))
