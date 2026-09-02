#!/usr/bin/env python3
"""Emit the state-doc results table from the two sources of truth: the draw index (measurements)
and docs/perf_baselines.json (the cells as committed). Nothing is typed by hand, so the doc cannot
drift from the file it describes. Run from the worktree."""
import json, pathlib, statistics

WT = pathlib.Path("/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap")
LEGS = ["rf3", "esmc-300m-single", "nesso1"]

draws: dict[str, list[float]] = {}
for r in (WT / "perf/qb2p300cgap/draws.tsv").read_text().splitlines():
    f = dict(p.split("=", 1) for p in r.split("\t") if "=" in p)
    if f.get("value") not in (None, "NA"):
        draws.setdefault(f["model"], []).append(float(f["value"]))

cells = json.loads((WT / "docs/perf_baselines.json").read_text())["cards"]["p300c"]["models"]

print("| leg | draws | band | median | new cell | cell vs median |")
print("|---|---|---|---|---|---|")
for m in LEGS:
    v = draws.get(m, [])
    if not v:
        print(f"| {m} | — | — | — | — | NOT DRAWN |")
        continue
    med = statistics.median(v)
    if m not in cells:
        print(f"| {m} | n={len(v)} | {(max(v)/min(v)-1)*100:.1f} % | {med:.6f} | NOT SEEDED | — |")
        continue
    cell, unit = cells[m]["value"], cells[m]["unit"]
    print(f"| {m} | n={len(v)} | {(max(v)/min(v)-1)*100:.1f} % | {med:.6f} | {cell:g} {unit} "
          f"| {(cell/med-1)*100:+.1f} % |")

print()
for m in LEGS:
    v = draws.get(m, [])
    print(f"{m}: " + " / ".join(f"{x:.6f}" for x in v))
