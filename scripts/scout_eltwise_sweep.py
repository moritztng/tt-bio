#!/usr/bin/env python3
"""Scout-only: is the 0.75 eltwise slowdown a fixed per-dispatch cost or a kernel regression?

Sweeps unary + binary eltwise across five tensor sizes. If the 0.75-minus-0.68 *absolute* gap
stays flat as the tensor grows, it is dispatch overhead (amortized by big tensors and by trace
replay). If the gap grows with size, it is a throughput regression and a hard blocker.

Also covers ttnn.add / ttnn.multiply -- tt-bio's two most-called ops (221 + 95 call sites), the
biggest gap in the earlier 13-op probe.

Usage: TT_VISIBLE_DEVICES=1 python3 scout_eltwise_sweep.py <out.json>
"""
import json, sys, time
import torch
import ttnn

OUT = sys.argv[1]

# (side, reps) -- fewer reps for the big tensors so a leg stays inside a few minutes.
SIZES = [(256, 200), (512, 200), (1024, 100), (2048, 50), (4096, 20), (8192, 10)]

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0)
import importlib.metadata as md
res = {"ttnn_version": md.version("ttnn"), "ops": {}}


def sync():
    ttnn.synchronize_device(dev)


def tt(x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def timeit(name, fn, reps):
    for _ in range(5):          # warm kernel cache + program cache before timing
        fn()
    sync()                      # drain: the timed region must start empty
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()                      # drain: queued device time must land inside the region
    ms = (time.perf_counter() - t0) * 1e3 / reps
    res["ops"][name] = {"ms_per_rep": ms, "reps": reps}
    print("  %-34s %9.4f ms  (n=%d)" % (name, ms, reps), flush=True)


UNARY = [("relu", ttnn.relu), ("sigmoid", ttnn.sigmoid),
         ("silu", ttnn.silu), ("gelu", ttnn.gelu)]

for side, reps in SIZES:
    a = tt(torch.randn(1, 1, side, side))
    b = tt(torch.randn(1, 1, side, side))
    for opname, op in UNARY:
        try:
            timeit("u_%s_%d" % (opname, side), lambda o=op, x=a: o(x), reps)
        except Exception:
            import traceback
            res["ops"]["u_%s_%d" % (opname, side)] = {"error": traceback.format_exc()}
            print("  u_%s_%d FAILED" % (opname, side), flush=True)
    for opname, op in (("add", ttnn.add), ("multiply", ttnn.multiply)):
        try:
            timeit("b_%s_%d" % (opname, side), lambda o=op, x=a, y=b: o(x, y), reps)
        except Exception:
            import traceback
            res["ops"]["b_%s_%d" % (opname, side)] = {"error": traceback.format_exc()}
            print("  b_%s_%d FAILED" % (opname, side), flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)

# Dispatch floor: smallest legal tile. Whatever this costs is pure host+dispatch, no real compute.
tiny = tt(torch.randn(1, 1, 32, 32))
timeit("floor_relu_32", lambda: ttnn.relu(tiny), 200)
timeit("floor_add_32", lambda: ttnn.add(tiny, tiny), 200)

ttnn.close_device(dev)
json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
failed = [n for n, d in res["ops"].items() if "error" in d]
print("wrote %s | ops=%d | failed=%d %s" % (OUT, len(res["ops"]), len(failed), failed))
