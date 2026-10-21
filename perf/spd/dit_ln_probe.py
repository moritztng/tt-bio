#!/usr/bin/env python3
"""Is the OpenFold3 DiT adaLN LayerNorm launch-bound? Interleaved vs block-sharded ttnn.layer_norm at [rows, 768].

The Wormhole fast census (pcen2, 1000 MHz) reads the two `_fuse_adaln` LayerNorms at 25 us a call, 9,600 calls a
fold (0.24 s), at 0.25 of the DRAM roof. Interleaved, ttnn spreads a LayerNorm over row tiles only: 512 rows are 16
tiles, so 16 of the chip's cores run it. Block-sharded, the row reduction spans a core row and every core holds a
[rows/gy, 768/gx] block. This prices that before a lever is built, traced us a call per arm:

  dram      interleaved DRAM in and out (the census shape)
  l1        interleaved L1 in and out (dit_l1)
  bs_GXxGY  block-sharded in and out on a GX x GY grid (the kernel alone)
  rt_GXxGY  L1 interleaved -> block-sharded -> LayerNorm -> L1 interleaved (what the block would pay today)

with rel. RMS error against a float64 LayerNorm. Random bf16 input; one chip; no model weights.

    python perf/spd/dit_ln_probe.py --out r.json
"""
import argparse, json, os, statistics, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
ap = argparse.ArgumentParser()
ap.add_argument("--rows", default="512")
ap.add_argument("--reps", type=int, default=200)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import torch
import ttnn
import tt_bio.tenstorrent as T

C = 768
dev = T.get_device(trace="diffusion")
torch.manual_seed(0)
g = dev.compute_with_storage_grid_size()
res = {"card": os.environ.get("TT_VISIBLE_DEVICES"), "host": os.uname().nodename, "arch": str(dev.arch()),
       "grid": str(g), "cells": []}
L1 = ttnn.L1_MEMORY_CONFIG


def ckc(fid):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=True, fp32_dest_acc_en=True,
                                            packer_l1_acc=True)


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


def sharded(M, gx, gy):
    mc = ttnn.create_sharded_memory_config(shape=(M, C), core_grid=ttnn.CoreGrid(y=gy, x=gx),
                                           strategy=ttnn.ShardStrategy.BLOCK,
                                           orientation=ttnn.ShardOrientation.ROW_MAJOR)
    bw, bh = C // 32 // gx, M // 32 // gy
    sw = max(d for d in (1, 2, 3, 4) if bw % d == 0)
    pc = ttnn.LayerNormShardedMultiCoreProgramConfig(compute_with_storage_grid_size=(gx, gy), subblock_w=sw,
                                                     block_h=bh, block_w=bw, inplace=False)
    return mc, pc


for M in (int(r) for r in a.rows.split(",")):
    xh = torch.randn(1, M, C) * 2.0 + 0.5
    ref = torch.nn.functional.layer_norm(xh.to(torch.bfloat16).double(), (C,), eps=1e-5)
    xd = ttnn.from_torch(xh.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    xl = ttnn.to_memory_config(xd, L1)
    for fname, fid in (("hifi4", ttnn.MathFidelity.HiFi4), ("hifi2", ttnn.MathFidelity.HiFi2)):
        ck = ckc(fid)
        arms = {"dram": lambda: ttnn.layer_norm(xd, epsilon=1e-5, compute_kernel_config=ck),
                "l1": lambda: ttnn.layer_norm(xl, epsilon=1e-5, compute_kernel_config=ck, memory_config=L1)}
        for gx, gy in ((8, 8), (8, 4), (6, 8), (4, 8)):
            if gx > g.x or gy > g.y or (C // 32) % gx or (M // 32) % gy:
                continue
            mc, pc = sharded(M, gx, gy)
            xs = ttnn.to_memory_config(xd, mc)
            arms[f"bs_{gx}x{gy}"] = (lambda xs=xs, mc=mc, pc=pc: ttnn.layer_norm(
                xs, epsilon=1e-5, compute_kernel_config=ck, program_config=pc, memory_config=mc))

            def rt(mc=mc, pc=pc):
                s = ttnn.to_memory_config(xl, mc)
                o = ttnn.layer_norm(s, epsilon=1e-5, compute_kernel_config=ck, program_config=pc, memory_config=mc)
                ttnn.deallocate(s)
                r = ttnn.to_memory_config(o, L1)
                ttnn.deallocate(o)
                return r
            arms[f"rt_{gx}x{gy}"] = rt
        for arm, fn in arms.items():
            c = {"M": M, "fid": fname, "arm": arm}
            try:
                o = fn()
                d = (ttnn.to_torch(ttnn.to_memory_config(o, ttnn.DRAM_MEMORY_CONFIG)).double().reshape(ref.shape)
                     - ref)
                ttnn.deallocate(o)
                c["max_abs"] = round(float(d.abs().max()), 5)
                c["rel_rms"] = round(float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()), 6)
                c["us"] = timed(fn, a.reps)
            except Exception as e:
                c["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            res["cells"].append(c)
            print(json.dumps(c), flush=True)
            a.out.write_text(json.dumps(res, indent=1))
res["done"] = time.strftime("%FT%TZ", time.gmtime())
a.out.write_text(json.dumps(res, indent=1))
T.cleanup()
