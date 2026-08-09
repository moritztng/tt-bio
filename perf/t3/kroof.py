#!/usr/bin/env python3
"""Re-measure the K-corrected COMPUTE roof without letting the DRAM write bound it.

Round 1 swept K with the output in DRAM. At M=10240 N=4096 that is 83.9 MB written per call, which at
this card's 273.4 GB/s write roof costs 307 us of the 430 us measured -- so the "K=256 roof of
49.95 TFLOP/s" was a write roof wearing a compute roof's label, and the Transition's fc2 beat it
(64.05 TFLOP/s). Redone with the output in L1, which is where fc1/fc2 actually put theirs, and swept
over N and over config so the reported number is the best rate this card reaches at that K.
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
CKC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=3, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o)


def T(shape, mc=DRAM):
    return ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                          device=dev, memory_config=mc)


res = {}
for K in (256, 384, 1024, 1536, 4096):
    best = {"tflops": 0.0}
    for M, N in ((2048, 2048), (4096, 2048), (4096, 4096), (8192, 2048), (10240, 1024)):
        try:
            a = T((1, 1, M, K), L1)
            b = T((1, 1, K, N), DRAM)
        except Exception as e:                                         # noqa: BLE001
            print(f"  K={K} M={M} N={N} alloc ERR {str(e)[:60]}", flush=True)
            continue
        gf = 2 * M * K * N / 1e9
        for lbl, kw in (("default", {}), ("core_grid", {"core_grid": CORE_GRID_MAIN})):
            try:
                s = timed(lambda: ttnn.deallocate(
                    ttnn.matmul(a, b, compute_kernel_config=CKC, memory_config=L1, **kw)))
            except Exception as e:                                     # noqa: BLE001
                print(f"  K={K:<5} M={M:<6} N={N:<5} {lbl:9s} ERR {str(e)[:50]}", flush=True)
                continue
            tf = gf / s / 1e3
            print(f"  K={K:<5} M={M:<6} N={N:<5} {lbl:9s} {s * 1e6:9.1f} us {tf:8.2f} TFLOP/s",
                  flush=True)
            if tf > best["tflops"]:
                best = {"tflops": round(tf, 2), "M": M, "N": N, "cfg": lbl, "us": round(s * 1e6, 1)}
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    res[K] = best
    print(f"  >>> K={K} L1-OUTPUT COMPUTE ROOF {best['tflops']} TFLOP/s "
          f"({best.get('cfg')}, M={best.get('M')} N={best.get('N')})\n", flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=1)
print("wrote " + sys.argv[1], flush=True)
