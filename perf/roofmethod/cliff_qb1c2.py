#!/usr/bin/env python3
"""What actually triggers ttnn 0.67.4's default-config collapse: the L1 OUTPUT, or L1 OPERANDS?

T3 framed the 12x cliff as an L1-output effect. On qb1 the sweep disagrees: with both inputs in L1
the default collapses to ~12 TFLOP/s with EITHER output (K=4096 nt=64: 12.20 oL1 / 12.24 oDRAM),
while the square probe, whose inputs are in DRAM, collapses only with an L1 output (4096^3: 128.13
oDRAM / 5.52 oL1). Those two runs differ in the input buffer type, so the axis is confounded.
This is the 2x2 that separates them, at fixed shape.
"""
import json, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
dg = dev.compute_with_storage_grid_size()
CKC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)


def timed(fn, warm=2, pipe=3, reps=5):
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


def T(shape, mc):
    return ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=dev, memory_config=mc)


rows = []
for name, M, K, N in (("K4096_nt64", 4608, 4096, 2048), ("square4096", 4096, 4096, 4096),
                      ("K256_nt32", 16384, 256, 1024), ("K256_nt8", 16384, 256, 256)):
    gf = 2 * M * K * N / 1e9
    for in_lbl, imc in (("iDRAM", DRAM), ("iL1", L1)):
        try:
            a, b = T((1, 1, M, K), imc), T((1, 1, K, N), imc)
        except Exception as e:                                               # noqa: BLE001
            print(f"  {name} {in_lbl} alloc ERR {str(e)[:60]}", flush=True)
            continue
        for out_lbl, omc in (("oDRAM", DRAM), ("oL1", L1)):
            for cfg_lbl, kw in (("default", {}), ("cg_13x10", {"core_grid": ttnn.CoreGrid(y=dg.y, x=dg.x)})):
                try:
                    s = timed(lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=CKC,
                                                                  memory_config=omc, **kw)))
                except Exception as e:                                       # noqa: BLE001
                    print(f"  {name:11s} {in_lbl} {out_lbl} {cfg_lbl:9s} ERR {str(e)[:45]}", flush=True)
                    continue
                tf = gf / s / 1e3
                rows.append({"shape": name, "M": M, "K": K, "N": N, "inputs": in_lbl,
                             "out": out_lbl, "cfg": cfg_lbl, "us": round(s * 1e6, 2),
                             "tflops": round(tf, 2)})
                print(f"  {name:11s} {in_lbl} {out_lbl} {cfg_lbl:9s} {s*1e6:9.2f} us {tf:8.2f} TFLOP/s",
                      flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)

json.dump(rows, open(sys.argv[1], "w"), indent=1)
print("wrote " + sys.argv[1], flush=True)
