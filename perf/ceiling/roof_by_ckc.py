#!/usr/bin/env python3
"""Which compute roof does the trunk actually get to aim at?

The production Pairformer runs HiFi4 with fp32_dest_acc_en=True and packer_l1_acc=True.
The 137.1 TFLOP/s figure in circulation was measured with packer_l1_acc=False. Those are
different roofs, so measure all of them on ONE card, same shapes, DRAM result (an L1
result thrashes above N=2048 and produces nonsense).
"""
import json
import statistics as st
import sys
import time

import torch

import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG


def timed(dev, fn, warm=4, pipe=6, reps=5):
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
        o.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(o)


dev = get_device()
g = dev.compute_with_storage_grid_size()
print(f"grid {g.x}x{g.y} = {g.x*g.y} cores", flush=True)
CFG = {
    "HiFi4_fp32acc_packer(production)": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True),
    "HiFi4_plain": dict(math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=False, packer_l1_acc=False),
    "HiFi2_plain": dict(math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=False, packer_l1_acc=False),
    "LoFi_plain": dict(math_fidelity=ttnn.MathFidelity.LoFi, fp32_dest_acc_en=False, packer_l1_acc=False),
}
res = {"grid": [g.x, g.y], "cores": g.x * g.y, "runs": {}}
torch.manual_seed(0)
for n in (4096, 6144):
    a = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, 1, n, n), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    gf = 2 * n ** 3 / 1e9
    for lbl, kw in CFG.items():
        kc = ttnn.init_device_compute_kernel_config(dev.arch(), **kw)
        try:
            ms = timed(dev, lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=kc, memory_config=DRAM)))
        except Exception as e:
            print(f"  N={n} {lbl} ERR {str(e)[:60]}", flush=True)
            continue
        tf = gf / (ms / 1e3) / 1e3
        res["runs"][f"{n}_{lbl}"] = {"ms": round(ms, 4), "tflops": round(tf, 2)}
        print(f"  N={n:<5} {lbl:34s} {ms:9.4f} ms {tf:8.2f} TFLOP/s", flush=True)
    ttnn.deallocate(a)
    ttnn.deallocate(b)
json.dump(res, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], flush=True)
