#!/usr/bin/env python3
"""How every model/rung sits against the fp32-softmax two-refusal retirement.

`FP32_SOFTMAX_L1_GRID` in each recorded cell carries served=l1_blocks, declined=l1_refused,
and `_FP32_SOFTMAX_FREE_REFUSAL_CAP` is 2: the second refusal retires the floating plan for
that class.

Reaching the cap is NOT by itself bad, and that is the trap in reading this column.
Retirement has two landing places and they differ by 20%:

  onto the tuned block, where the 8x8 rectangle still serves the shape. MEASURED at
  openbind/768: the shipped 768 KB takes both refusals and is the FASTEST arm, 87.9 s
  against 91.2 s at 384 KB and 93.1 s at 512 KB -- both of which take zero refusals and
  hold nearly twice the blocks resident (ab_openbind_768.json). Buying residency loses here.

  onto nothing, where the rectangle is dark. MEASURED at openbind/1280 forced to 896 KB:
  439.3 s against 363.0 s, l1 calls 440 -> 2 (ab_openbind_1280.json).

`served` after the cap is what separates them: a large block count means it landed on the
tuned block, a count near zero means it landed on nothing.

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
            rows.append((m, int(rung), g.get("served"), g.get("declined")))

if not rows:
    print(f"no {CARD} cell carries FP32_SOFTMAX_L1_GRID counters")
    raise SystemExit(0)

print(f"{'model':14s} {'rung':>5s} {'l1_blocks':>10s} {'refused':>8s}  left  reading")
hot = []
for m, rung, served, declined in rows:
    d = declined or 0
    left = CAP - d
    if served == 0:
        flag = "path dark at this shape"
    elif left <= 0 and served > 1000:
        flag = "retired ONTO THE TUNED BLOCK (measured faster, not a defect)"
    elif left <= 0:
        flag = "retired ONTO NOTHING: this is the 1.21x cliff"
    elif left == 1:
        flag = "ONE refusal left"
        hot.append((m, rung))
    else:
        flag = ""
    print(f"{m:14s} {rung:5d} {served:10d} {d:8d}  {left:>4d}  {flag}")

print()
if hot:
    where = ", ".join(f"{m}/{r}" for m, r in hot)
    print("One refusal from retirement, so these are the shapes a budget change tips first."
          f" Which way it tips depends on whether the tuned block is alive there: {where}")
else:
    print("no cell sits one refusal from retirement on the recorded engine")
print("Counters come from the recorded cells, taken at a stale engine: this says where to"
      " look, not what the margin is today.")
