#!/usr/bin/env python3
"""Scout-only: characterise the 0.75 fixed-cost-per-op floor found by the size sweep.

The sweep showed 0.68 and 0.75 converge exactly at 4096/8192 but differ ~9x on a 32x32 op,
i.e. 0.75 carries a fixed per-op cost that is independent of tensor size. Three questions:

  A. how big is the floor, measured the same way on both versions (sync-bracketed);
  B. is it spent on the HOST (time to return from the call) or on the DEVICE (only visible
     at the drain)? -- host_ms is the loop with no sync inside, total_ms adds the drain;
  C. does TRACE REPLAY erase it? tt-bio already traces its hot loops, so if the floor is
     host-dispatch and trace removes it, the regression is largely irrelevant to us.

Usage: TT_VISIBLE_DEVICES=1 python3 scout_dispatch_probe.py <out.json>
"""
import json, sys, time
import torch
import ttnn

OUT = sys.argv[1]
CHAIN = 50          # ops per trace body
REPS = 100

torch.manual_seed(0)
dev = ttnn.open_device(device_id=0, trace_region_size=200_000_000)
import importlib.metadata as md
res = {"ttnn_version": md.version("ttnn"), "chain": CHAIN, "reps": REPS, "m": {}}


def sync():
    ttnn.synchronize_device(dev)


def tt(x):
    return ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def split_timing(name, fn, reps):
    """host_ms = time to issue; total_ms = issue + drain. Their gap is device-side time."""
    for _ in range(5):
        fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    t1 = time.perf_counter()
    sync()
    t2 = time.perf_counter()
    res["m"][name] = {"host_ms": (t1 - t0) * 1e3 / reps,
                      "total_ms": (t2 - t0) * 1e3 / reps, "reps": reps}
    print("  %-26s host %8.4f  total %8.4f ms/op" %
          (name, res["m"][name]["host_ms"], res["m"][name]["total_ms"]), flush=True)


x512 = tt(torch.randn(1, 1, 512, 512))
y512 = tt(torch.randn(1, 1, 512, 512))
x32 = tt(torch.randn(1, 1, 32, 32))

split_timing("eager_relu_32", lambda: ttnn.relu(x32), REPS)
split_timing("eager_relu_512", lambda: ttnn.relu(x512), REPS)
split_timing("eager_add_512", lambda: ttnn.add(x512, y512), REPS)

# --- C. trace replay: same CHAIN of ops, host dispatch cost paid once at capture ---
try:
    # warm the program cache with the exact ops the trace body will contain
    for _ in range(3):
        t = x512
        for _ in range(CHAIN):
            t = ttnn.relu(t)
    sync()

    tid = ttnn.begin_trace_capture(dev, cq_id=0)
    t = x512
    for _ in range(CHAIN):
        t = ttnn.relu(t)
    ttnn.end_trace_capture(dev, tid, cq_id=0)
    sync()

    for _ in range(3):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    sync()

    n = 20
    t0 = time.perf_counter()
    for _ in range(n):
        ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    t1 = time.perf_counter()
    sync()
    t2 = time.perf_counter()
    res["m"]["traced_relu_512"] = {
        "host_ms": (t1 - t0) * 1e3 / (n * CHAIN),
        "total_ms": (t2 - t0) * 1e3 / (n * CHAIN), "reps": n * CHAIN}
    print("  %-26s host %8.4f  total %8.4f ms/op" %
          ("traced_relu_512", res["m"]["traced_relu_512"]["host_ms"],
           res["m"]["traced_relu_512"]["total_ms"]), flush=True)
    ttnn.release_trace(dev, tid)
except Exception:
    import traceback
    res["m"]["traced_relu_512"] = {"error": traceback.format_exc()}
    print("  traced_relu_512 FAILED", flush=True)
    print(res["m"]["traced_relu_512"]["error"], flush=True)

ttnn.close_device(dev)
json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
print("wrote", OUT)
