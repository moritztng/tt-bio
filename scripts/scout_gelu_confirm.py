#!/usr/bin/env python3
"""Scout-only: confirm or kill the `gelu` large-size outlier from the 0.68->0.75 eltwise sweep.

The sweep (38 legs) found every op converging to ~1.00x at 4096/8192 -- consistent with the
regression being a fixed per-dispatch host cost -- EXCEPT gelu, which came out 1.25x/1.26x
slower on 0.75 at n=20/n=10 reps. One op refusing to converge either means the "large ops are
unaffected" conclusion has a hole in it, or it was noise.

This script is the confirming experiment. Three design points that the sweep did not have:

  * CONTROLS AT IDENTICAL SIZES. relu/silu/sigmoid are timed at the same 4096 and 8192 in the
    same round as gelu. A gelu-specific ratio is only meaningful against ops that converge.
  * INTERLEAVED WITHIN A ROUND. Ops rotate inside each round, so drift and thermals hit all of
    them equally instead of accumulating on whichever op is timed last.
  * REPEATED ROUNDS -> A STATED NOISE FLOOR. The sweep reported a single n=20/n=10 point per op
    with no spread, so 1.25x was not falsifiable. Here each op gets ROUNDS independent timed
    legs; the per-op spread across rounds IS the noise floor, and a ratio inside it is not a
    finding.

Reps are raised well above the sweep's (50 at 4096, 25 at 8192) because a warm eltwise cache
makes each timed leg only milliseconds of device time -- the sweep's low rep counts were budgeted
for a 38-leg cold run, not for resolving a 25% claim.

Usage: TT_VISIBLE_DEVICES=1 python3 scout_gelu_confirm.py <out.json>
"""
import json, statistics, sys, time
import torch
import ttnn

OUT = sys.argv[1]
ROUNDS = 5
SIZES = [(4096, 50), (8192, 25)]

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
import importlib.metadata as md
res = {"ttnn_version": md.version("ttnn"), "rounds": ROUNDS, "legs": {}}

OPS = [("gelu", ttnn.gelu), ("relu", ttnn.relu),
       ("silu", ttnn.silu), ("sigmoid", ttnn.sigmoid)]


def sync():
    ttnn.synchronize_device(dev)


# One resident input per size: allocation is not part of what we are timing.
inputs = {side: ttnn.from_torch(torch.randn(side, side), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)
          for side, _ in SIZES}


def timed(fn, reps):
    for _ in range(5):          # warm kernel + program cache
        fn()
    sync()                      # drain: the timed region must start empty
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()                      # drain: queued device time must land inside the region
    return (time.perf_counter() - t0) * 1e3 / reps


# Warm every leg once before any round is recorded, so round 1 is not a cold outlier.
for side, reps in SIZES:
    for _, op in OPS:
        timed(lambda o=op, x=inputs[side]: o(x), 3)

for r in range(ROUNDS):
    for side, reps in SIZES:
        for name, op in OPS:                     # ops rotate inside the round
            ms = timed(lambda o=op, x=inputs[side]: o(x), reps)
            res["legs"].setdefault("%s_%d" % (name, side), []).append(ms)
            print("  round %d  %-14s %9.4f ms  (n=%d)" % (r + 1, "%s_%d" % (name, side), ms, reps),
                  flush=True)

print("\n=== per-leg median and noise floor (%d rounds) ===" % ROUNDS)
for leg, vals in res["legs"].items():
    med = statistics.median(vals)
    spread = (max(vals) / min(vals) - 1) * 100
    res.setdefault("summary", {})[leg] = {"median_ms": med, "min_ms": min(vals),
                                          "max_ms": max(vals), "spread_pct": spread}
    print("%-14s median %9.4f ms  min %9.4f  max %9.4f  spread %+6.1f%%"
          % (leg, med, min(vals), max(vals), spread))

json.dump(res, open(OUT, "w"), indent=1)
ttnn.close_device(dev)
