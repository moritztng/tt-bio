#!/usr/bin/env python3
"""Emit the state-doc results table from the two sources of truth: the draw index (measurements)
and docs/perf_baselines.json (the cells as committed). Nothing here is typed by hand, so the doc
cannot drift from the file it describes. Run from the worktree."""
import json, pathlib, statistics, sys

WT = pathlib.Path("/home/ttuser/.coworker/wt/qb2-card-layer-baseline-reseed")

# Old cells, as they stood before this task, read out of the parent task's committed table
# (state/qb2-new-hardware-baseline-crosscheck.md, "The other half"), plus the retired-box control:
# the v0.7.2 release gate of 2026-08-28, which ran this protocol on the previous physical qb2.
OLD = {"boltz2": 1.497659, "esmfold2": 2.413904, "esmfold2-fast": 3.125835,
       "esmc-300m": 33.169045, "esmc-600m": 22.095491, "esmc-6b": 4.749028,
       "boltzgen": 0.017057, "boltz2-affinity": 0.014319}
CONTROL = {"boltz2": 1.7632, "esmfold2": 3.2347, "esmfold2-fast": 3.8323,
           "esmc-300m": 150.4193, "esmc-600m": 107.3617, "esmc-6b": None,
           "boltzgen": 0.021410, "boltz2-affinity": 0.02294}

draws: dict[str, list[float]] = {}
for r in (WT / "perf/qb2cardlayer/draws.tsv").read_text().splitlines():
    f = dict(p.split("=", 1) for p in r.split("\t") if "=" in p)
    if f.get("value") not in (None, "NA"):
        draws.setdefault(f["model"], []).append(float(f["value"]))

cells = json.loads((WT / "docs/perf_baselines.json").read_text())["cards"]["p300c"]["models"]

def pct(a, b):
    return f"{(a / b - 1) * 100:+.1f} %"

print("| leg | old cell | draws | median | new cell | Δ median vs cell | retired box (v0.7.2) "
      "| Δ median vs box | verdict |")
print("|---|---|---|---|---|---|---|---|---|")
for m in OLD:
    v = draws.get(m, [])
    if not v:
        print(f"| {m} | {OLD[m]:g} | — | — | — | — | — | — | NOT DRAWN |")
        continue
    med = statistics.median(v)
    cell = cells[m]["value"]
    unit = cells[m]["unit"]
    ctl = CONTROL[m]
    reseeded = cell != OLD[m]
    print(f"| {m} | {OLD[m]:g} {unit} | n={len(v)}, band {(max(v)/min(v)-1)*100:.1f} % "
          f"| {med:.6f} | {cell:g} | {pct(med, OLD[m])} "
          f"| {ctl if ctl else 'none (UNCONTROLLED)'} | {pct(med, ctl) if ctl else '—'} "
          f"| {'RESEEDED' if reseeded else 'NOT RESEEDED'} |")

# Step 4 of the brief: does the parent task's own reading still hold today? Its numbers are a prior
# data point, not a floor — if a leg had failed to reproduce, the honest verdict was
# CONFIRMED-STALE-NOT-REPRODUCING and no reseed. These are the parent's live medians from
# state/qb2-new-hardware-baseline-crosscheck.md, "The other half".
PARENT = {"boltz2": 1.743458, "esmfold2": 3.249686, "esmfold2-fast": 3.9505,
          "esmc-300m": 146.990257, "esmc-600m": 109.928138, "esmc-6b": 10.469379,
          "boltzgen": 0.021287, "boltz2-affinity": 0.023563}
print()
print("| leg | parent task median (2026-09-01) | this task median | Δ | reproduces? |")
print("|---|---|---|---|---|")
for m, p in PARENT.items():
    med = statistics.median(draws[m])
    d = (med / p - 1) * 100
    print(f"| {m} | {p:g} | {med:.6f} | {d:+.1f} % | {'yes' if abs(d) < 6 else 'NO'} |")
