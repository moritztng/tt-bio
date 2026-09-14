#!/usr/bin/env python3
"""The shipped default, not a monkeypatch: what `_FP32_SOFTMAX_L1_GRID` reads after device open.

`TT_BIO_FP32_SOFTMAX_L1_LIVE_GRID=0` pins the fitted rectangle, so the two arms here are the two
branches of the change itself. They cannot be interleaved in one process (the flag is read at
import), so each arm reports its own served-block census beside its time and the ratio is quoted
against the interleaved sweep, not against the other process.
"""
import json, os, sys, time
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
cc = dev.compute_with_storage_grid_size()
kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
        else ttnn.types.BlackholeComputeKernelConfig)
kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
          fp32_dest_acc_en=True, packer_l1_acc=True)
torch.manual_seed(0)   # both processes build the same operands, so the digests are comparable
S, heads, hd = 512, 4, 32
D = ttnn.DRAM_MEMORY_CONFIG
mk = lambda *sh: ttnn.from_torch(torch.randn(*sh, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                                 device=dev, memory_config=D)
q, k, v = mk(S, heads, S, hd), mk(S, heads, S, hd), mk(S, heads, S, hd)
bias = mk(1, heads, S, S)
call = lambda: T._fp32_softmax_attention(q, k, v, bias, scale_inv=float(hd) ** -0.5,
                                         compute_kernel_config=kc, out_dtype=ttnn.bfloat16,
                                         bias_scale_inv=1.0)
ttnn.deallocate(call())
for key in T.FP32_SOFTMAX_STATS:
    T.FP32_SOFTMAX_STATS[key] = 0
best = float("inf")
for _ in range(4):
    t0 = time.perf_counter()
    for _ in range(2):
        ttnn.deallocate(call())
    ttnn.synchronize_device(dev)
    best = min(best, (time.perf_counter() - t0) / 2)
out = {"live_grid_flag": T._FP32_SOFTMAX_L1_LIVE_GRID,
       "grid": list(T._FP32_SOFTMAX_L1_GRID), "compute_grid": [cc.x, cc.y],
       "is_small_grid": T._IS_SMALL_GRID, "ms": best * 1e3,
       "stats": dict(T.FP32_SOFTMAX_STATS),
       "digest": float(ttnn.to_torch(call()).float().abs().sum())}
print(json.dumps(out), flush=True)
T.cleanup()
