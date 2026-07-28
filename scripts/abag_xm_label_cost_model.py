#!/usr/bin/env python3
"""Cost the label phase from measured labels, priced per target by size.

Written because two of my own schedule estimates were badly wrong in the same way. Both applied a flat
median per label to a fold count. That fails twice over:

  * the median used was a WALL measured while several label workers ran concurrently, not a serial
    cost. Dividing that by the worker count double-counts the contention already in it;
  * label cost is strongly superlinear in target size -- fitted exponent ~2.2, so cost grows ~4.7x
    when a target doubles -- because the 1225-pair DockQ matrix dominates and each pair scales with
    structure size. A flat median is wrong in both directions depending on which targets are left.

Fits wall = a * tokens^b by least squares on log(wall) vs log(tokens) over every completed label in
this host's campaign logs, then prices the unlabelled backlog by each fold's own token count.

    abag_xm_label_cost_model.py [--slab]     # --slab also prices the whole 492-fold slab

Reports the fit residual, because an estimate without an error bar is what got this wrong twice.
"""
import argparse
import glob
import json
import math
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SLICES = ROOT / "docs" / "implementation-parity-data" / "abag-xm-tier-a-slices.json"
TIER = Path.home() / "abag_xm" / "tier_a"
LOGS = Path.home() / "abag_xm" / "logs"
GEN_DIR = {"protenix-v2": "protenix_v2", "opendde-abag": "opendde_abag", "boltz2": "boltz2"}

ap = argparse.ArgumentParser()
ap.add_argument("--slab", action="store_true", help="also price all 164 targets x 3 generators")
ap.add_argument("--workers", type=int, default=None,
                help="label workers on this host (default: nproc/8, matching the supervisor)")
a = ap.parse_args()

tokens = json.load(open(SLICES))["tokens"]

points = []
for f in glob.glob(str(LOGS / "labels_campaign.log*")):
    for ln in open(f, errors="ignore"):
        m = re.search(r"\[label\] (\S+) \S+ status=ok n=\d+ wall=([\d.]+)s", ln)
        if m and tokens.get(m.group(1)):
            points.append((tokens[m.group(1)], float(m.group(2))))
if len(points) < 8:
    raise SystemExit(f"only {len(points)} completed labels with token counts; too few to fit")

xs = [math.log(t) for t, _ in points]
ys = [math.log(w) for _, w in points]
n = len(xs)
mx, my = sum(xs) / n, sum(ys) / n
b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
aa = math.exp(my - b * mx)
rms = math.sqrt(sum((y - (math.log(aa) + b * x)) ** 2 for x, y in zip(xs, ys)) / n)


def predict(tok):
    return aa * tok ** b


import os
workers = a.workers if a.workers else max(1, (os.cpu_count() or 8) // 8)
print(f"fit over {n} completed labels: wall = {aa:.3e} * tokens^{b:.2f}")
print(f"  log-residual RMS {rms:.3f} (~x{math.exp(rms):.2f} typical per-fold factor error)")
print(f"  a target twice the size costs ~{2 ** b:.1f}x")
print(f"  sanity: observed median {statistics.median([w for _, w in points]):.0f}s vs fitted "
      f"{statistics.median([predict(t) for t, _ in points]):.0f}s")

recs = [json.loads(l) for l in open(TIER / "progress.jsonl") if l.strip()]
seen, backlog = set(), []
for r in recs:
    if r.get("status") != "ok":
        continue
    k = (r["target"], r["model"])
    if k in seen:
        continue
    seen.add(k)
    lab = TIER / "labels" / f"{GEN_DIR[r['model']]}_{r['target']}.json"
    tok = tokens.get(r["target"])
    if not lab.exists() and tok:
        backlog.append(tok)

cost = sum(predict(t) for t in backlog)
print(f"\nbacklog on this host: {len(backlog)} folds, tokens median "
      f"{statistics.median(backlog):.0f} (max {max(backlog)})" if backlog else "\nbacklog: none")
if backlog:
    print(f"  {cost / 3600:.1f} core-hours -> {cost / workers / 3600:.1f} h at {workers} workers")

if a.slab:
    total = sum(predict(t) for t in tokens.values()) * 3
    print(f"\nwhole slab: {len(tokens)} targets x 3 = {len(tokens) * 3} folds")
    print(f"  {total / 3600:.1f} core-hours total label cost")
    print(f"  measured so far on this host: {sum(w for _, w in points) / 3600:.1f} core-hours "
          f"over {n} labels")
