#!/usr/bin/env python3
"""Exit 0 when a leg is finished: it has a cards.p300c.models cell AND at least N drawn values.

Both halves matter. A cell with too few draws behind it is a number with no noise floor, and draws
with no cell are a measurement nobody can gate on. Used by campaign.sh so a relaunched chain
resumes at the first unfinished leg instead of redrawing a leg that is already done.
"""
import json, pathlib, sys

model, need = sys.argv[1], int(sys.argv[2])
O = pathlib.Path("/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap")
cells = json.loads((O / "docs/perf_baselines.json").read_text())["cards"]["p300c"]["models"]
idx = O / "perf/qb2p300cgap/draws.tsv"
n = 0
if idx.exists():
    for r in idx.read_text().splitlines():
        f = dict(p.split("=", 1) for p in r.split("\t") if "=" in p)
        if f.get("model") == model and f.get("value") not in (None, "NA"):
            n += 1
sys.exit(0 if model in cells and n >= need else 1)
