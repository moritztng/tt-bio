#!/usr/bin/env python3
"""The matmul roof at TriangleMultiplication's OWN shapes, not at 4096 cubed.

The 512 aa floor derived trimul at 5.64 s against 51.98 TFLOP/s, but that roof was measured on a
square 4096^3 matmul. Trimul does not run square matmuls. It runs, per call at padded N=512 and
pair width c:

  6 projections   [N*N, c] x [c, c]          -- 262144 x 256 x 256 in the trunk. Tall and skinny:
                                                K = c = 256, so each output tile reads a whole
                                                256-deep operand strip for 256 MACs of depth.
  1 contraction   batch c of [N, N] x [N, N]  -- 256 batched 512x512x512.

A tall-skinny matmul cannot reach a square-matmul roof, so 51.98 TFLOP/s flatters the residual. This
measures the roof each of those two shapes actually has, which is what the floor has to be built on.

Reuses wh-perf-esmfold2's bench/ckc approach; the arms are HiFi4 + fp32 acc, the config the pair
track runs, on the model's own main grid.
"""
import argparse, json, os, statistics as st, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--tree", type=Path, required=True)
ap.add_argument("--N", type=int, default=512)
ap.add_argument("--out", type=Path, required=True)
a = ap.parse_args()
sys.path.insert(0, str(a.tree.resolve()))

import torch
import ttnn
import tt_bio.tenstorrent as T


def bench(fn, n=5, warm=2):
    dev = T.get_device()
    for _ in range(warm):
        o = fn(); ttnn.synchronize_device(dev)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        o = fn(); ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if isinstance(o, ttnn.Tensor):
            ttnn.deallocate(o)
    return st.median(ts) * 1e3


def ckc(fid="HiFi4", fp32acc=True):
    cls = (ttnn.types.WormholeComputeKernelConfig if T.is_wormhole()
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=getattr(ttnn.MathFidelity, fid),
               math_approx_mode=False, fp32_dest_acc_en=fp32acc, packer_l1_acc=True)


dev = T.get_device()
g = dev.compute_with_storage_grid_size()
GRID = T.CORE_GRID_MAIN
N = a.N
R = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
     "grid": [g.x, g.y], "core_grid_main": list(T.COMPUTE_GRID_MAIN), "N": N, "roofs": {}}
f = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

for c in (256, 64):
    # 1. the projection shape: [N*N, c] x [c, c]
    try:
        A = f(torch.randn(1, N * N, c)); W = f(torch.randn(1, c, c))
        flop = 2 * (N * N) * c * c
        for fid in ("HiFi4", "LoFi"):
            ms = bench(lambda: ttnn.matmul(A, W, compute_kernel_config=ckc(fid), core_grid=GRID,
                                           dtype=ttnn.bfloat16))
            R["roofs"][f"proj_{N*N}x{c}x{c}_{fid}"] = round(flop / (ms * 1e-3) / 1e12, 3)
            R["roofs"][f"proj_{N*N}x{c}x{c}_{fid}_ms"] = round(ms, 4)
        ttnn.deallocate(A); ttnn.deallocate(W)
    except Exception as e:                                                    # noqa: BLE001
        R["roofs"][f"proj_c{c}"] = f"ERR {type(e).__name__}: {str(e)[:120]}"

    # 2. the contraction shape: batch c of [N, N] x [N, N]
    try:
        Ab = f(torch.randn(c, N, N)); Bb = f(torch.randn(c, N, N))
        flop = 2 * c * N ** 3
        for fid in ("HiFi4", "LoFi"):
            ms = bench(lambda: ttnn.matmul(Ab, Bb, compute_kernel_config=ckc(fid), core_grid=GRID,
                                           dtype=ttnn.bfloat16), n=3, warm=1)
            R["roofs"][f"contract_b{c}_{N}cubed_{fid}"] = round(flop / (ms * 1e-3) / 1e12, 3)
            R["roofs"][f"contract_b{c}_{N}cubed_{fid}_ms"] = round(ms, 4)
        ttnn.deallocate(Ab); ttnn.deallocate(Bb)
    except Exception as e:                                                    # noqa: BLE001
        R["roofs"][f"contract_c{c}"] = f"ERR {type(e).__name__}: {str(e)[:120]}"

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1), flush=True)
T.cleanup()
