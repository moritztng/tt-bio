#!/usr/bin/env python3
"""Harvest per-design scRMSD from a reference BoltzGen analysis CSV into results.json."""
import csv, json, statistics as st
from pathlib import Path

HERE = Path(__file__).parent
df = list(csv.DictReader(open(HERE / "aggregate_metrics_analyze.csv")))
col = "designfolding-bb_rmsd"
seqcol = "designed_sequence"
rows = []
vals = []
for r in df:
    v = float(r[col])
    vals.append(v)
    ln = len(r[seqcol]) if seqcol in r and r[seqcol] else None
    rows.append({"id": r["id"], "len": ln, "scrmsd": v})
out = {
    "metric": "designfolding-bb_rmsd (scRMSD, isolated refold, Kabsch CA-RMSD in Angstrom)",
    "column": col,
    "n": len(rows),
    "rows": rows,
    "min": min(vals),
    "median": st.median(vals),
    "mean": st.mean(vals),
    "stdev": st.pstdev(vals) if len(vals) > 1 else 0.0,
    "pass_le_2A": sum(1 for v in vals if v <= 2.0) / len(vals),
    "pass_le_4A": sum(1 for v in vals if v <= 4.0) / len(vals),
}
(HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
