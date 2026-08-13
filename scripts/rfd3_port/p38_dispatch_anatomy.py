"""p38 — what is a ttnn enqueue actually made of?

The host ledger (p35) charges RFD3 83.0 ms/step to ttnn enqueue over 3418 ops, ~24 us each, and
§1.4 turns that into a 122.6 ms/step "hard host floor". That floor is the single biggest object in
the step now, and nothing has ever decomposed the 24 us. Three candidates, and they have completely
different levers:

  A. fixed host cost per op   (pybind + program-cache hash + runtime-arg write) -> only fewer ops
  B. output allocation        (every op allocates a fresh DRAM buffer)          -> buffer reuse
  C. command-queue backpressure (the enqueue blocks because the CQ is full)     -> NOT host at all;
                                 the "host floor" would be hidden device time and the floor is wrong

This probe separates them with no model code:
  1. per-op enqueue with the device idle and the queue empty  -> A (+B)
  2. same op with a preallocated output tensor                -> A alone, so B = 1 - 2
  3. the same tiny op enqueued right behind a long device op  -> C
  4. enqueue cost vs tensor size / core count                 -> is A per-core runtime args?

Run: TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half python3 p38_dispatch_anatomy.py
"""
import json
import os
import statistics
import sys
import time

import torch
import ttnn

OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/p38_dispatch_anatomy.json"
REPS = int(os.environ.get("P38_REPS", "300"))


def med(xs):
    return statistics.median(xs) * 1e6  # us


def timed_enqueues(fn, reps, warm=8):
    """Call fn() reps times, timing each call. No sync inside the loop."""
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r) if isinstance(r, ttnn.Tensor) else None
    t_drain0 = time.perf_counter()
    ttnn.synchronize_device(dev)
    drain = time.perf_counter() - t_drain0
    return ts, drain


def dram(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT):
    return ttnn.from_torch(t, dtype=dtype, layout=layout, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


res = {"host": os.uname().nodename, "loadavg": os.getloadavg()}
dev = ttnn.open_device(device_id=0)
ttnn.enable_program_cache(dev) if hasattr(ttnn, "enable_program_cache") else None
try:
    # ---------------------------------------------------------------- 1+2+4
    # eltwise add at four sizes: tiny, DiT-ish, atom-row, and the 45 M-element monster
    cases = [
        ("tiny_32x32", (1, 1, 32, 32), ttnn.bfloat16),
        ("dit_256x768", (1, 1, 256, 768), ttnn.bfloat16),
        ("dit_250x768", (1, 1, 250, 768), ttnn.bfloat16),
        ("atom_3359x128", (1, 4, 3359, 128), ttnn.bfloat16),
        ("monster_3359x3360_f32", (1, 4, 3359, 3360), ttnn.float32),
    ]
    res["add"] = {}
    for name, shape, dt in cases:
        a = dram(torch.randn(*shape), dtype=dt)
        b = dram(torch.randn(*shape), dtype=dt)
        c = dram(torch.zeros(*shape), dtype=dt)
        n = 40 if name.startswith("monster") else REPS
        ts, drain = timed_enqueues(lambda: ttnn.add(a, b), n)
        ts_pre, drain_pre = timed_enqueues(lambda: ttnn.add(a, b, output_tensor=c), n)
        res["add"][name] = {
            "shape": list(shape), "dtype": str(dt), "reps": n,
            "alloc_us": med(ts), "prealloc_us": med(ts_pre),
            "alloc_min_us": min(ts) * 1e6, "prealloc_min_us": min(ts_pre) * 1e6,
            "drain_ms": drain * 1e3, "drain_pre_ms": drain_pre * 1e3,
            "device_ms_per_op": drain * 1e3 / n,
        }
        print(name, json.dumps(res["add"][name]), flush=True)
        for t in (a, b, c):
            ttnn.deallocate(t)

    # ---------------------------------------------------------------- op-type sweep at DiT shape
    x = dram(torch.randn(1, 1, 256, 768))
    w = dram(torch.randn(1, 1, 768, 768))
    g = dram(torch.randn(1, 1, 1, 768))
    ops = {
        "linear_256x768x768": lambda: ttnn.linear(x, w),
        "multiply": lambda: ttnn.multiply(x, x),
        "sigmoid": lambda: ttnn.sigmoid(x),
        "rms_norm": lambda: ttnn.rms_norm(x, weight=g),
        "typecast": lambda: ttnn.typecast(x, ttnn.float32),
        "reshape": lambda: ttnn.reshape(x, (1, 256, 768)),
        "permute": lambda: ttnn.permute(x, (0, 1, 3, 2)),
        "softmax": lambda: ttnn.softmax(x, dim=-1),
        "clone": lambda: ttnn.clone(x),
    }
    res["ops"] = {}
    for name, fn in ops.items():
        try:
            ts, drain = timed_enqueues(fn, REPS)
            res["ops"][name] = {"enqueue_us": med(ts), "min_us": min(ts) * 1e6,
                                "device_us_per_op": drain * 1e6 / REPS}
        except Exception as e:
            res["ops"][name] = {"error": str(e)[:200]}
        print(name, json.dumps(res["ops"][name]), flush=True)

    # ---------------------------------------------------------------- 3. backpressure
    # A long device op (the 45 M-element fp32 clone, ~0.94 ms measured) followed by tiny enqueues.
    # If the tiny enqueue costs the same as it did with an idle device, the CQ is not the limiter
    # and the ledger's dispatch time is real host work.
    big_a = dram(torch.randn(1, 4, 3359, 3360), dtype=ttnn.float32)
    ta = dram(torch.randn(1, 1, 256, 768))
    ttnn.synchronize_device(dev)
    burst = []
    for _ in range(30):
        r = ttnn.clone(big_a)          # ~0.9 ms of device
        row = []
        for _ in range(12):
            t0 = time.perf_counter()
            q = ttnn.multiply(ta, ta)  # ~10 us of device
            row.append((time.perf_counter() - t0) * 1e6)
            ttnn.deallocate(q)
        ttnn.deallocate(r)
        burst.append(row)
    ttnn.synchronize_device(dev)
    flat = [v for row in burst for v in row]
    res["backpressure"] = {
        "tiny_behind_big_us_median": statistics.median(flat),
        "tiny_behind_big_us_p90": sorted(flat)[int(0.9 * len(flat))],
        "by_position_us": [statistics.median([row[i] for row in burst]) for i in range(12)],
    }
    print("backpressure", json.dumps(res["backpressure"]), flush=True)
    ttnn.deallocate(big_a)
    ttnn.deallocate(ta)

    # ---------------------------------------------------------------- 5. deallocate / allocate cost
    z = [dram(torch.randn(1, 1, 256, 768)) for _ in range(64)]
    t0 = time.perf_counter()
    for t in z:
        ttnn.deallocate(t)
    res["deallocate_us"] = (time.perf_counter() - t0) * 1e6 / 64
    print("deallocate_us", res["deallocate_us"], flush=True)
finally:
    ttnn.close_device(dev)

with open(OUT, "w") as f:
    json.dump(res, f, indent=2)
print("wrote", OUT)
