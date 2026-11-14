#!/usr/bin/env python3
"""Can a generic_op program carry ESMFold2's token-DiT linears? minimal_matmul vs ttnn.linear at the DiT's shapes.

A persistent program for a DiT sub-block can only re-drive matmul kernels the wheel ships, and the only one tt-bio
has forked is `minimal_matmul` (mm_generic.py). At the pair SwiGLU shape it was ~6.5x off the multicast matmul
`ttnn.linear` picks (swiglu_fused.py), which killed that fusion; ESMC on Wormhole runs it by default above 2048 rows.
This measures where the token DiT sits: for each (rows, K, N) the DiT issues, traced us a call of

  lin     ttnn.linear on CORE_GRID_MAIN, ESMFold2's `_lin`
  mm      ttnn.experimental.minimal_matmul, default config
  mmc     the same with ESMC's block config (`esmc._mm_config`)
  mmf     dit_minimal_matmul_addcmul_fused (out = resid + 1.0 * (x @ w) * gate): the matmul with the DiT's gated
          residual folded in, only where N = 768

at HiFi4 + fp32 accumulation (the TorchWrapper default) and at HiFi2, with max |diff| against a float64 x @ w.
Random bf16 operands at unit scale; one chip; no model weights.

    python perf/spd/dit_mm_probe.py --out r.json
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="384,512,1920,2560")
ap.add_argument("--reps", type=int, default=50)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T
from tt_bio import esmc

dev = T.get_device(trace="diffusion")
torch.manual_seed(0)
# (K, N): adaLN/out_gate/q 768x768, folded q|k|v 768x3072 (heads padded 48->64), g 768x1024, padded out_proj
# 1024x768, swish 768x6144, lin_out 3072x768
SHAPES = [(768, 768), (768, 3072), (768, 1024), (1024, 768), (768, 6144), (3072, 768)]
res = {"card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "arch": str(dev.arch()),
       "grid": str(dev.compute_with_storage_grid_size()), "cells": []}


def ckc(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False, fp32_dest_acc_en=True,
                                            packer_l1_acc=True)


def up(t):
    return ttnn.from_torch(t.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)


def timed(fn, n):
    for _ in range(2):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    tid, out = T.capture_trace(dev, lambda: [fn() for _ in range(10)])
    for t in out:
        ttnn.deallocate(t)
    ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
    ts = []
    for _ in range(3):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        for _ in range(n // 10):
            ttnn.execute_trace(dev, tid, cq_id=0, blocking=False)
        ttnn.synchronize_device(dev); ts.append((time.perf_counter() - t0) / (n // 10 * 10) * 1e6)
    ttnn.release_trace(dev, tid)
    return round(statistics.median(ts), 2)


for M in (int(r) for r in a.rows.split(",")):
    for K, N in SHAPES:
        xh, wh = torch.randn(1, M, K), torch.randn(K, N) / K ** 0.5
        rh, gh = torch.randn(1, M, N), torch.rand(1, N)
        x, w, r, g = up(xh), up(wh), up(rh), up(gh)
        ref = xh.to(torch.bfloat16).double() @ wh.to(torch.bfloat16).double()
        reff = rh.to(torch.bfloat16).double() + ref * gh.to(torch.bfloat16).double()
        for fname, fid in (("hifi4", ttnn.MathFidelity.HiFi4), ("hifi2", ttnn.MathFidelity.HiFi2)):
            ck = ckc(fid)
            g_ = dev.compute_with_storage_grid_size()
            cfg = esmc._mm_config(x, w) or {"config": ttnn.MinimalMatmulConfig(
                M_block_size=16, K_block_size=4 if (K // 32) % 4 == 0 else 2, N_block_size=4, subblock_h=2,
                subblock_w=2, compute_with_storage_grid_size=ttnn.CoreCoord(g_.x, g_.y))}
            arms = {
                "lin": (lambda: ttnn.linear(x, w, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                            core_grid=T.CORE_GRID_MAIN), ref),
                "mm": (lambda: ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                                                memory_config=ttnn.DRAM_MEMORY_CONFIG), ref),
                "mmc": (lambda: ttnn.experimental.minimal_matmul(x, w, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG, **cfg), ref),
            }
            if N == 768:
                arms["mmf"] = (lambda: ttnn.experimental.dit_minimal_matmul_addcmul_fused(
                    x, w, 1.0, r, g, compute_kernel_config=ck, dtype=ttnn.bfloat16,
                    memory_config=ttnn.DRAM_MEMORY_CONFIG, **cfg), reff)
            for arm, (fn, rf) in arms.items():
                c = {"M": M, "K": K, "N": N, "fid": fname, "arm": arm}
                try:
                    o = fn()
                    d = (ttnn.to_torch(o).double().reshape(rf.shape) - rf).abs()
                    ttnn.deallocate(o)
                    c["max_abs"] = round(float(d.max()), 5)
                    c["rel_rms"] = round(float(d.pow(2).mean().sqrt() / rf.pow(2).mean().sqrt()), 6)
                    c["us"] = timed(fn, a.reps)
                except Exception as e:
                    c["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                res["cells"].append(c)
                print(json.dumps(c), flush=True)
                a.out.write_text(json.dumps(res, indent=1))
        for t in (x, w, r, g):
            ttnn.deallocate(t)
res["done"] = time.strftime("%FT%TZ", time.gmtime())
a.out.write_text(json.dumps(res, indent=1))
T.cleanup()
