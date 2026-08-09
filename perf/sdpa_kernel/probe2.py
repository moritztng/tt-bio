#!/usr/bin/env python3
"""W9 probe 2: a real DRAM read roof, the head_dim question re-run without W6's confound,
and an honest bfp8_b-bias accuracy number.

W6's head_dim sweep ran with NO program config, so every point sat on ttnn's default
chunking. This card measures 3.809 ms for d=32/b=320 without bias under that default but
0.644 ms with q_chunk=k_chunk=320. The default chunking is therefore a large confound and
W6's "head_dim=32 costs 1.59x" needs re-running at the full-length config the model now
uses. Prediction if the one-tile-contraction mechanism is real and independent of chunking:
the 1.59x survives. If it collapses, the 1.59x was a chunking artefact.

W6's bfp8 comparison in probe 1 was invalid (q/k/v were redrawn between the two cases).
Here the same q/k/v tensors serve both bias dtypes.
"""
import argparse
import json
import time

import torch
import ttnn

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="perf/sdpa_kernel/probe2_qb2c1.json")
args = ap.parse_args()

DEV = ttnn.open_device(device_id=0)
CKC = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
    fp32_dest_acc_en=True, packer_l1_acc=True)
DRAM = ttnn.DRAM_MEMORY_CONFIG
SEQ = 320


def timed(fn, iters=5, amort=4):
    for _ in range(2):
        r = fn()
        if r is not None:
            ttnn.deallocate(r)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(DEV)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(amort)]
        ttnn.synchronize_device(DEV)
        ts.append((time.perf_counter() - t0) * 1e3 / amort)
        for o in outs:
            if o is not None:
                ttnn.deallocate(o)
    ts.sort()
    return ts[len(ts) // 2]


res = {"card": "qb2-card1", "read_roof": {}, "headdim": [], "bfp8": {}}

# --------------------------------------------------- a read-only DRAM roof
# ttnn.sum over the last dim reads the whole tensor and writes 1/N of it.
print("== DRAM read roof (reduction: reads everything, writes ~nothing) ==", flush=True)
for mb in (52, 105, 210):
    rows = mb * 1_000_000 // (4096 * 2) // 32 * 32
    t = ttnn.from_torch(torch.randn(1, 1, rows, 4096), dtype=ttnn.bfloat16,
                        layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=DRAM)
    nb = rows * 4096 * 2
    ms = timed(lambda: ttnn.sum(t, dim=-1), amort=2)
    gb = nb / ms * 1e-6
    print(f"sum over {nb/1e6:6.1f} MB: {ms:7.4f} ms  {gb:7.1f} GB/s read", flush=True)
    res["read_roof"][f"sum_{mb}mb_gbs"] = gb
    ttnn.deallocate(t)


def sdpa(q, k, v, bias, d, qc, kc):
    prog = ttnn.SDPAProgramConfig(
        compute_with_storage_grid_size=DEV.compute_with_storage_grid_size(),
        q_chunk_size=qc, k_chunk_size=kc, exp_approx_mode=False)
    return ttnn.transformer.scaled_dot_product_attention(
        q, k, v, attn_mask=bias, is_causal=False, scale=d ** -0.5,
        program_config=prog, compute_kernel_config=CKC, memory_config=DRAM)


# --------------------------------------------------- head_dim, full-length chunking
print("\n== head_dim sweep at q_chunk=k_chunk=320 (b*h*d held at 81920, seq 320) ==",
      flush=True)
print(f"{'b':>5} {'h':>3} {'d':>5} {'bias':>6} {'ms':>9} {'TFLOP/s':>8} {'read GB/s':>10}")
BH = 320 * 8 * 32
for d in (32, 64, 128, 256):
    b = BH // (8 * d)
    torch.manual_seed(11)
    q, k, v = (ttnn.from_torch(torch.randn(b, 8, SEQ, d) * 0.3, dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=DRAM)
               for _ in range(3))
    bias = ttnn.from_torch(torch.randn(1, 8, SEQ, SEQ) * 0.3, dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=DEV, memory_config=DRAM)
    for name, bt in (("bf16", bias), ("none", None)):
        ms = timed(lambda: sdpa(q, k, v, bt, d, SEQ, SEQ))
        flops = 4 * b * 8 * SEQ * SEQ * d
        rd = 3 * b * 8 * SEQ * d * 2 + (b * 8 * SEQ * SEQ * 2 if bt is not None else 0)
        print(f"{b:>5} {8:>3} {d:>5} {name:>6} {ms:9.4f} {flops/ms*1e-9:8.1f} "
              f"{rd/ms*1e-6:10.1f}", flush=True)
        res["headdim"].append(dict(b=b, h=8, d=d, bias=name, ms=ms,
                                   tflops=flops / ms * 1e-9, read_gbs=rd / ms * 1e-6))
    for t in (q, k, v, bias):
        ttnn.deallocate(t)

# --------------------------------------------------- bfp8_b bias, same q/k/v
print("\n== bfp8_b bias: perf and accuracy against the identical bf16 run ==", flush=True)
B, H, D = 320, 8, 32
torch.manual_seed(23)
qt, kt, vt = (torch.randn(B, H, SEQ, D) * 0.3 for _ in range(3))
bt_ = torch.randn(1, H, SEQ, SEQ) * 0.3
q, k, v = (ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                           device=DEV, memory_config=DRAM) for x in (qt, kt, vt))
outs = {}
for name, dt in (("bf16", ttnn.bfloat16), ("bfp8_b", ttnn.bfloat8_b)):
    bias = ttnn.from_torch(bt_, dtype=dt, layout=ttnn.TILE_LAYOUT, device=DEV,
                           memory_config=DRAM)
    ms = timed(lambda: sdpa(q, k, v, bias, D, SEQ, SEQ))
    o = sdpa(q, k, v, bias, D, SEQ, SEQ)
    outs[name] = ttnn.to_torch(o).float()
    ttnn.deallocate(o)
    ttnn.deallocate(bias)
    res["bfp8"][name + "_ms"] = ms
    print(f"{name:>7} bias: {ms:.4f} ms", flush=True)
a, c = outs["bf16"], outs["bfp8_b"]
rmsd = (a - c).pow(2).mean().sqrt().item()
std = a.std().item()
pcc = torch.corrcoef(torch.stack([a.flatten(), c.flatten()]))[0, 1].item()
res["bfp8"].update(rmsd=rmsd, std=std, rmsd_over_std=rmsd / std, pcc=pcc,
                   speedup=res["bfp8"]["bf16_ms"] / res["bfp8"]["bfp8_b_ms"])
print(f"rmsd/std {rmsd/std:.5f}  PCC {pcc:.6f}  speedup "
      f"{res['bfp8']['speedup']:.3f}x  (W6 noise band on z: 0.0185-0.0217)")

for t in (q, k, v):
    ttnn.deallocate(t)
json.dump(res, open(args.out, "w"), indent=1)
print(f"\nwrote {args.out}")
ttnn.close_device(DEV)
