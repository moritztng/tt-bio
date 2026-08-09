#!/usr/bin/env python3
"""W9 probe 3: bfp8_b bias -- perf and accuracy, with q/k/v held identical.

The bias re-read is 77% of SDPA's DRAM read traffic, so halving the bias element size
should cut the op time by the bias term alone. Prediction from the marginal bandwidth
measured in probe 1/2 (421-451 GB/s on the bias stream): 0.647 ms (no-bias floor) +
262.1 MB / 421 GB/s = 1.27 ms, i.e. 1.42x. Accuracy is the gate: W6's shipped SDPA
reduction-order band on the block output is rmsd/std 0.0185-0.0217.

Also checks bfloat4_b, and the bias-permute cost at each dtype, since the same tensor
feeds the [1,8,320,320] permute that W7 ranks second in the whole model.
"""
import json
import time

import torch
import ttnn

DEV = ttnn.open_device(device_id=0)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
B, H, N, D = 320, 8, 320, 32


def timed(fn, iters=5, amort=4):
    for _ in range(2):
        ttnn.deallocate(fn())
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(amort)]
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3 / amort)
        for o in outs:
            ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2]


prog = ttnn.SDPAProgramConfig(
    compute_with_storage_grid_size=DEV.compute_with_storage_grid_size(),
    q_chunk_size=N, k_chunk_size=N, exp_approx_mode=False)

torch.manual_seed(23)
qt, kt, vt = (torch.randn(B, H, N, D) * 0.3 for _ in range(3))
biast = torch.randn(1, H, N, N) * 0.3
q, k, v = (ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=DEV, memory_config=DRAM) for x in (qt, kt, vt))

res = {"card": "qb2-card1", "cases": {}}
ref = None
print(f"{'bias dtype':>12} {'ms':>9} {'speedup':>8} {'rmsd/std':>10} {'PCC':>10}")
for name, dt in (("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b),
                 ("bfloat4_b", ttnn.bfloat4_b)):
    try:
        bias = ttnn.from_torch(biast, dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV,
                               memory_config=DRAM)
    except Exception as e:
        print(f"{name:>12}  from_torch failed: {str(e).splitlines()[0][:60]}")
        continue

    def call():
        return ttnn.transformer.scaled_dot_product_attention(
            q, k, v, attn_mask=bias, is_causal=False, scale=D ** -0.5,
            program_config=prog, compute_kernel_config=CKC, memory_config=DRAM)

    try:
        ms = timed(call)
    except Exception as e:
        print(f"{name:>12}  sdpa failed: {str(e).splitlines()[0][:60]}")
        ttnn.deallocate(bias)
        continue
    o = call()
    out = ttnn.to_torch(o).float()
    ttnn.deallocate(o)
    ttnn.deallocate(bias)
    if ref is None:
        ref = out
        rmsd = pcc = 0.0
        base = ms
    else:
        rmsd = (ref - out).pow(2).mean().sqrt().item() / ref.std().item()
        pcc = torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1].item()
    print(f"{name:>12} {ms:9.4f} {base/ms:8.3f} {rmsd:10.5f} {pcc:10.6f}", flush=True)
    res["cases"][name] = dict(ms=ms, speedup=base / ms, rmsd_over_std=rmsd, pcc=pcc)

# the bias permute at each dtype: [1,320,320,8] -> [1,8,320,320]
print("\n== the bias-building permute (0,3,1,2) at each dtype ==")
for name, dt in (("bfloat16", ttnn.bfloat16), ("bfloat8_b", ttnn.bfloat8_b)):
    try:
        src = ttnn.from_torch(torch.randn(1, N, N, H), dtype=dt, layout=ttnn.TILE_LAYOUT,
                              device=DEV, memory_config=DRAM)
        ms = timed(lambda: ttnn.permute(src, (0, 3, 1, 2)), amort=4)
        print(f"{name:>12} {ms:9.4f} ms")
        res.setdefault("bias_permute", {})[name] = ms
        ttnn.deallocate(src)
    except Exception as e:
        print(f"{name:>12}  failed: {str(e).splitlines()[0][:70]}")

json.dump(res, open("perf/sdpa_kernel/probe3_qb2c1.json", "w"), indent=1)
print("\nwrote perf/sdpa_kernel/probe3_qb2c1.json")
ttnn.close_device(DEV)
