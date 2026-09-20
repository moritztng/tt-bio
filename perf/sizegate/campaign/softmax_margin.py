#!/usr/bin/env python3
"""How close every model/rung sits to the fp32-softmax two-refusal cliff.

`FP32_SOFTMAX_L1_GRID` in each recorded cell carries served=l1_blocks, declined=l1_refused.
`_FP32_SOFTMAX_FREE_REFUSAL_CAP` is 2: the second refusal retires the floating plan for that
class, and where the tuned rectangle is dark it retires onto nothing. Measured cost of
landing there, openbind/1280 on card 1 at 1350 MHz: 439.3 s against 363.0 s, l1 calls
440 -> 2 (perf/sizegate/campaign/ab_openbind_1280.json).

So `declined` is not a curiosity, it is the distance to a 20% cliff, and it is already
recorded for every model. Read it before deciding whether the budget is an openbind knob or
a shared one.

Usage: softmax_margin.py [card]
"""
import glob
import json
import sys

CARD = sys.argv[1] if len(sys.argv) > 1 else "p150a"
CAP = 2
rows = []
for f in sorted(glob.glob("docs/size_ladder_baseline.d/*.json")):
    d = json.load(open(f))
    for m, ent in sorted((d.get("cards", {}).get(CARD, {}).get("models", {}) or {}).items()):
        for rung, lev in sorted((ent.get("levers") or {}).items(), key=lambda kv: int(kv[0])):
            g = (lev or {}).get("FP32_SOFTMAX_L1_GRID")
            if not isinstance(g, dict) or g.get("served") is None:
                continue
            rows.append((m, int(rung), g.get("served"), g.get("declined"), g.get("resolved")))

if not rows:
    print(f"no {CARD} cell carries FP32_SOFTMAX_L1_GRID counters")
    raise SystemExit(0)

print(f"{'model':14s} {'rung':>5s} {'l1_blocks':>10s} {'refused':>8s}  margin to the cap")
hot = []
for m, rung, served, declined, res in rows:
    d = declined or 0
    left = CAP - d
    flag = ""
    if served == 0:
        flag = "  path dark at this shape"
    elif left <= 0:
        flag = "  AT THE CAP: the plan is retired here"
    elif left == 1:
        flag = "  ONE refusal left"
        hot.append((m, rung))
    print(f"{m:14s} {rung:5d} {served:10d} {d:8d}  {left:>2d}{flag}")

print()
if hot:
    print("one refusal from the cliff, and therefore the shapes a budget change would tip"
          f" first: {', '.join(f'{m}/{r}' for m, r in hot)}")
else:
    print("no cell sits one refusal from the cap on the recorded engine")
print("Counters are from the recorded cells; those were taken at a stale engine, so treat"
      " this as where to look, not as the current margin.")
