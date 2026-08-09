#!/usr/bin/env python3
"""Why does a trunk matmul get 30 TFLOP/s when a square 4096 matmul gets 101?

Two candidate mechanisms, and they are separable by sweeping one dim at a time:
  (a) contraction depth. K=256 gives 8 tiles of reuse per output tile; a 4096 square gives 128.
  (b) DRAM traffic. At K=N=256 the op reads and writes 52.4 MB for 13.4 GFLOP, AI 128 FLOP/byte,
      under this card's ~261 machine balance, so it should be memory-bound and nowhere near the
      compute roof.
Sweeping K at fixed M,N moves both AI and reuse together, so also sweep with the operands and the
result pinned to L1, which removes DRAM from the picture entirely and leaves only (a).
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=3, pipe=4, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) * 1e3 / pipe)
    return st.median(out)


dev = get_device()
g = dev.compute_with_storage_grid_size()
print(f"grid {g.x}x{g.y} = {g.x*g.y} cores", flush=True)
KC = ttnn.init_device_compute_kernel_config(
    dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True
)
MM = ttnn.experimental.minimal_matmul
res = {"grid": [g.x, g.y], "k_sweep_dram": {}, "n_sweep_dram": {}, "k_sweep_l1": {}, "l1_copy_roof": {}}
torch.manual_seed(0)


def best_rate(a, b, m, k, n, mem):
    gf = 2 * m * k * n / 1e9
    cands = {}
    for name, fn in (
        ("minimal_matmul", lambda: ttnn.deallocate(MM(a, b, memory_config=mem))),
        ("matmul+grid", lambda: ttnn.deallocate(ttnn.matmul(a, b, compute_kernel_config=KC, memory_config=mem,
                                                            core_grid=ttnn.CoreGrid(x=g.x, y=g.y)))),
    ):
        try:
            ms = timed(dev, fn)
        except Exception:
            continue
        cands[name] = (ms, gf / (ms / 1e3) / 1e3)
    if not cands:
        return None
    nm = min(cands, key=lambda x: cands[x][0])
    return nm, cands[nm][0], cands[nm][1]


M = 102400  # the pair track's M at 298 aa: 320 rows x 320 cols
print("== K sweep, M=102400 N=256, operands+result in DRAM ==", flush=True)
for k in (32, 64, 128, 256, 512, 1024, 2048):
    a = ttnn.from_torch(torch.randn(1, 1, M, k) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    b = ttnn.from_torch(torch.randn(1, 1, k, 256) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    r = best_rate(a, b, M, k, 256, DRAM)
    if r:
        nm, ms, tf = r
        rw = (M * k + M * 256) * 2 / 1e9
        res["k_sweep_dram"][k] = {"backend": nm, "ms": round(ms, 4), "tflops": round(tf, 2),
                                  "rw_GB": round(rw, 4), "GBs": round(rw / (ms / 1e3), 1),
                                  "AI": round(2 * M * k * 256 / (rw * 1e9), 1)}
        print(f"  K={k:5d} {nm:15s} {ms:8.4f} ms {tf:7.2f} TFLOP/s  {rw/(ms/1e3):7.1f} GB/s  AI={2*M*k*256/(rw*1e9):6.1f}", flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)

print("== N sweep, M=102400 K=256, DRAM ==", flush=True)
a = ttnn.from_torch(torch.randn(1, 1, M, 256) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
for n in (32, 64, 128, 256, 512, 1024):
    b = ttnn.from_torch(torch.randn(1, 1, 256, n) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    r = best_rate(a, b, M, 256, n, DRAM)
    if r:
        nm, ms, tf = r
        rw = (M * 256 + M * n) * 2 / 1e9
        res["n_sweep_dram"][n] = {"backend": nm, "ms": round(ms, 4), "tflops": round(tf, 2),
                                  "GBs": round(rw / (ms / 1e3), 1), "AI": round(2 * M * 256 * n / (rw * 1e9), 1)}
        print(f"  N={n:5d} {nm:15s} {ms:8.4f} ms {tf:7.2f} TFLOP/s  {rw/(ms/1e3):7.1f} GB/s  AI={2*M*256*n/(rw*1e9):6.1f}", flush=True)
    ttnn.deallocate(b)
ttnn.deallocate(a)

# Same K sweep with everything in L1, at an M small enough that operands+result fit.
# M=8192 -> at K=1024 that is 16 MB in, 4 MB out, well inside the 110-core L1.
print("== K sweep, M=8192 N=256, operands+result in L1 (DRAM removed) ==", flush=True)
ML = 8192
for k in (32, 64, 128, 256, 512, 1024, 2048):
    try:
        a = ttnn.from_torch(torch.randn(1, 1, ML, k) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
        b = ttnn.from_torch(torch.randn(1, 1, k, 256) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
    except Exception as e:
        print(f"  K={k} alloc ERR {str(e)[:60]}", flush=True)
        continue
    r = best_rate(a, b, ML, k, 256, L1)
    if r:
        nm, ms, tf = r
        res["k_sweep_l1"][k] = {"backend": nm, "ms": round(ms, 4), "tflops": round(tf, 2)}
        print(f"  K={k:5d} {nm:15s} {ms:8.4f} ms {tf:7.2f} TFLOP/s", flush=True)
    ttnn.deallocate(a); ttnn.deallocate(b)

# L1 copy roof at the pair-tensor chunk shapes, for the memory-side floor
print("== L1<->L1 clone roof ==", flush=True)
for shape in [(1, 320, 320, 64), (1, 320, 320, 128), (1, 320, 320, 256)]:
    try:
        t = ttnn.from_torch(torch.randn(*shape) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
        ms = timed(dev, lambda: ttnn.deallocate(ttnn.clone(t, memory_config=L1)))
        byt = 1
        for d in shape:
            byt *= d
        byt *= 2
        res["l1_copy_roof"][str(shape)] = {"ms": round(ms, 5), "MB": round(byt / 1e6, 2),
                                           "GBs_rw": round(2 * byt / 1e9 / (ms / 1e3), 1)}
        print(f"  {shape} {byt/1e6:7.2f} MB  {ms:8.5f} ms  {2*byt/1e9/(ms/1e3):7.1f} GB/s (read+write)", flush=True)
        ttnn.deallocate(t)
    except Exception as e:
        print(f"  {shape} ERR {str(e)[:70]}", flush=True)

json.dump(res, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], flush=True)
