#!/usr/bin/env python3
"""Why is `batched_matmul` 20-45% slower than the identical config passed explicitly?

Only two things sit between them: the guard chain (two `memory_config()` pybind calls) and
`_batched_matmul_config`, which calls `ttnn.get_max_worker_l1_unreserved_size()` on EVERY call.
A device query that is cheap on an idle device can be a pipeline drain when work is queued, and
this loop queues 6 calls before it synchronises.

Arms: naive; batched_matmul; batched_matmul with the chooser monkeypatched to return a
precomputed config (no device query, guard chain kept); the config passed explicitly.

Prediction: if the L1 query drains the queue, arm `hoisted` lands on `explicit`; if it does not,
`hoisted` stays with `main` and the cost is the guard chain itself.
"""
import json, statistics as st, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

H, NQ, NK, DH = 4, 32, 128, 32
REPS, TRIALS, WARM = 6, 7, 3

dev = T.get_device()
CKC = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
GRID = tuple(T.COMPUTE_GRID_MAIN)
L1 = int(ttnn.get_max_worker_l1_unreserved_size())
print(f"grid {GRID}, L1 {L1}", flush=True)


def timed(fn):
    for _ in range(WARM):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / REPS


def tt(x, dt):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)


rows = []
for nb in (75, 110):
    g = torch.Generator().manual_seed(0)
    q = tt(torch.randn(nb, H, NQ, DH, generator=g), ttnn.float32)
    kt = tt(torch.randn(nb, H, DH, NK, generator=g), ttnn.float32)
    a = tt(torch.rand(nb, H, NQ, NK, generator=g), ttnn.float32)
    v = tt(torch.randn(nb, H, NK, DH, generator=g), ttnn.float32)
    for lbl, x, y, kt_n, nt_n in (("QK^T", q, kt, 1, 4), ("A@V", a, v, 4, 1)):
        cfg = T._batched_matmul_search(nb * H, 1, kt_n, nt_n, 4, GRID, L1)
        real_chooser = T._batched_matmul_config

        def hoisted():
            T._batched_matmul_config = lambda *a, **k: cfg
            try:
                return T.batched_matmul(x, y, compute_kernel_config=CKC)
            finally:
                T._batched_matmul_config = real_chooser

        arms = {
            "naive": lambda: ttnn.matmul(x, y, compute_kernel_config=CKC),
            "main": lambda: T.batched_matmul(x, y, compute_kernel_config=CKC),
            "hoisted": hoisted,
            "explicit": lambda: ttnn.matmul(x, y, program_config=cfg, compute_kernel_config=CKC),
        }
        ref = ttnn.to_torch(arms["naive"]())
        exact = {k: bool(torch.equal(ttnn.to_torch(f()), ref)) for k, f in arms.items()}
        names = list(arms)
        samples = {k: [] for k in names}
        for t in range(TRIALS):
            for k in names[t % len(names):] + names[:t % len(names)]:
                samples[k].append(timed(arms[k]))
        us = {k: round(st.median(v) * 1e6, 2) for k, v in samples.items()}
        rows.append({"nb": nb, "op": lbl, "us": us, "bit_exact": exact})
        print(f"nb={nb:3d} {lbl:5s} naive={us['naive']:8.2f} main={us['main']:8.2f} "
              f"hoisted={us['hoisted']:8.2f} explicit={us['explicit']:8.2f} | exact={exact}",
              flush=True)
    for t in (q, kt, a, v):
        ttnn.deallocate(t)

# Is the L1 query a pipeline drain? Price it with the queue empty and with 6 matmuls queued.
g = torch.Generator().manual_seed(0)
x = tt(torch.randn(75, H, NQ, DH, generator=g), ttnn.float32)
y = tt(torch.randn(75, H, DH, NK, generator=g), ttnn.float32)
ttnn.synchronize_device(dev)
t0 = time.perf_counter()
for _ in range(200):
    ttnn.get_max_worker_l1_unreserved_size()
idle_us = (time.perf_counter() - t0) / 200 * 1e6
busy = []
for _ in range(20):
    for _ in range(6):
        ttnn.deallocate(ttnn.matmul(x, y, compute_kernel_config=CKC))
    t0 = time.perf_counter()
    ttnn.get_max_worker_l1_unreserved_size()
    busy.append((time.perf_counter() - t0) * 1e6)
    ttnn.synchronize_device(dev)
print(f"get_max_worker_l1_unreserved_size: idle {idle_us:.3f} us, "
      f"with 6 matmuls queued median {st.median(busy):.3f} us", flush=True)

out = sys.argv[1] if len(sys.argv) > 1 else "perf/atomwindow_reconcile/probe3_qb1c0.json"
json.dump({"rows": rows, "l1_query_idle_us": round(idle_us, 3),
           "l1_query_queued_us": round(st.median(busy), 3)}, open(out, "w"), indent=2)
print("wrote", out, flush=True)
