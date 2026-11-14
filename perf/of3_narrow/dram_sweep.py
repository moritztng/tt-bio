#!/usr/bin/env python3
"""The pairformer attention pair-bias projection [1, S, S, 128] x [128, 16] when its normed input sits in DRAM
(Wormhole at 512 tokens: the z LayerNorm does not fit L1, so `_narrow_proj_linear` reads DRAM with the
L1-tuned config and the WH census reads 1053 us/call against 125 us on a p150a, 8.4x for a 2.06x chip).

Arms: `shipped` (T._narrow_proj_linear, DRAM leg), `auto` (ttnn.linear, no config), `grid` (core_grid=),
and 1D-mcast configs over in0_block_w x out_block_h x per_core_M. Each arm: median us over --iters
synchronised calls and torch.equal against `shipped`. in0_block_w = 1 keeps the shipped K blocking, so by
the repo's measured rule (tenstorrent._attn_value_program_config) an equal arm is a bit-exact lever.
Usage: TT_VISIBLE_DEVICES=<chip> timeout 900 python perf/of3_narrow/dram_sweep.py --out r.json [--n 512]
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch, ttnn
from tt_bio import tenstorrent as T

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=512)
ap.add_argument("--iters", type=int, default=15)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = T.get_device()
S, K, N = a.n, 128, 16
torch.manual_seed(0)
X = ttnn.from_torch(torch.randn(1, S, S, K).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG)
W = ttnn.from_torch((torch.randn(K, N) / K ** 0.5).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
ckc = ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi3, math_approx_mode=False,
                                       fp32_dest_acc_en=True, packer_l1_acc=True)
gx, gy = T.COMPUTE_GRID_MAIN
m_tiles, k_tiles, n_tiles = S * (S // 32), K // 32, 1


def cfg(bw, obh, pcm, cores=(gx, gy)):
    sh = max(h for h in range(min(4, obh), 0, -1) if obh % h == 0)
    return ttnn.MatmulMultiCoreReuseMultiCast1DProgramConfig(
        compute_with_storage_grid_size=cores, in0_block_w=bw, out_subblock_h=sh, out_subblock_w=1,
        out_block_h=obh, out_block_w=n_tiles, per_core_M=pcm, per_core_N=n_tiles, fuse_batch=True,
        fused_activation=None, mcast_in0=False)


arms = {"shipped": lambda: T._narrow_proj_linear(X, W, ckc, ttnn.bfloat16),
        "auto": lambda: ttnn.linear(X, W, compute_kernel_config=ckc, dtype=ttnn.bfloat16,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG),
        "grid": lambda: ttnn.linear(X, W, compute_kernel_config=ckc, dtype=ttnn.bfloat16, core_grid=T.CORE_GRID_MAIN,
                                    memory_config=ttnn.DRAM_MEMORY_CONFIG)}
nc = gx * gy
pcm0 = -(-m_tiles // nc)
for pcm in sorted({pcm0, -(-pcm0 // 5) * 5, -(-pcm0 // 8) * 8}):
    for bw in (1, 2, 4):
        for obh in sorted({d for d in (1, 2, 4, 5, 8, 10, 16, 20, 23, pcm) if pcm % d == 0}):
            arms[f"c_bw{bw}_obh{obh}_pcm{pcm}"] = (lambda bw=bw, obh=obh, pcm=pcm: ttnn.linear(
                X, W, compute_kernel_config=ckc, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG,
                program_config=cfg(bw, obh, pcm)))

res, ref = {}, None
for name, fn in arms.items():
    try:
        y = fn(); ttnn.synchronize_device(dev)
        t = ttnn.to_torch(y).float(); ttnn.deallocate(y)
        if not torch.isfinite(t).all():
            raise RuntimeError("non-finite output")
        ref = t if name == "shipped" else ref
        us = []
        for _ in range(a.iters):
            ttnn.synchronize_device(dev); t0 = time.perf_counter()
            y = fn(); ttnn.synchronize_device(dev); us.append((time.perf_counter() - t0) * 1e6); ttnn.deallocate(y)
        res[name] = {"us": round(statistics.median(us), 1), "equal": bool(torch.equal(t, ref)),
                     "max_abs_vs_shipped": float((t - ref).abs().max())}
    except Exception as e:                                                      # noqa: BLE001
        res[name] = {"error": str(e).splitlines()[0][:200]}
    print(name, res[name], flush=True)
json.dump({"n": S, "arch": T.arch_name(), "grid": [gx, gy], "arms": res}, open(a.out, "w"), indent=1)
print("wrote", a.out)
