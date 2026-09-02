#!/usr/bin/env python3
"""Move one model's freshly written baseline from the machine block down to the card block.

perf_regression.py --update-baseline always writes to cards.<card>.machines.<machine>.models, but
these three cells belong in the card-level fallback cards.p300c.models: p300c has exactly one
machine, and a card-layer cell is where the other eleven p300c cells of the same kind live. Writing
with the tool and then moving the entry keeps the number tool-measured, never hand-typed, and keeps
exactly one home for it. A copy left behind in the machine block would shadow the card cell, which
is the dead-cell shape this same branch just deleted for protenix-v2 and opendde.
"""
import json, sys, pathlib

model, = sys.argv[1:]
p = pathlib.Path("/home/ttuser/.coworker/wt/p300c-baseline-coverage-gap/docs/perf_baselines.json")
d = json.loads(p.read_text())
card = d["cards"]["p300c"]
mach = card["machines"]["tt-quietbox2"]["models"]
if model not in mach:
    sys.exit(f"{model}: nothing new in the machine block, --update-baseline did not run")
old = card["models"].get(model, {}).get("value")
card["models"][model] = mach.pop(model)
new = card["models"][model]["value"]
p.write_text(json.dumps(d, indent=2) + "\n")
print(f"{model}: cards.p300c.models {old} -> {new} ({(new/old-1)*100:+.1f}%)" if old
      else f"{model}: cards.p300c.models seeded {new}")
