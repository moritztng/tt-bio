#!/usr/bin/env python3
"""The roof at TriangleAttention's own shapes, the companion to wh_shape_roof.py.

Per call at padded N=512 and pair width c, with head_dim 32 so h = c/32 heads:

  qkv projection   [N*N, c] x [c, 3c]
  gate projection  [N*N, c] x [c, c]
  out projection   [N*N, c] x [c, c]
  bias projection  [N*N, c] x [c, h]        -- tiny, measured anyway so the model has no hole
  SDPA             q,k,v [N, h, N, 32]      -- attention within each row of the pair tensor

Same reason as the trimul probe: the 4096-cubed square roof is 6.4x optimistic for these shapes and
a floor built on it is meaningless.
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
    h = max(1, c // 32)
    for tag, nout in (("qkv", 3 * c), ("gate", c), ("out", c), ("bias", max(32, h))):
        try:
            A = f(torch.randn(1, N * N, c)); W = f(torch.randn(1, c, nout))
            flop = 2 * (N * N) * c * nout
            ms = bench(lambda: ttnn.matmul(A, W, compute_kernel_config=ckc(), core_grid=GRID,
                                           dtype=ttnn.bfloat16))
            R["roofs"][f"{tag}_c{c}_x{nout}_ms"] = round(ms, 4)
            R["roofs"][f"{tag}_c{c}_x{nout}_TFLOPs"] = round(flop / (ms * 1e-3) / 1e12, 3)
            ttnn.deallocate(A); ttnn.deallocate(W)
        except Exception as e:                                                # noqa: BLE001
            R["roofs"][f"{tag}_c{c}"] = f"ERR {type(e).__name__}: {str(e)[:110]}"

    # SDPA at the tri-att shape: one attention per row of the pair tensor.
    try:
        q = f(torch.randn(N, h, N, 32)); k = f(torch.randn(N, h, N, 32)); v = f(torch.randn(N, h, N, 32))
        flop = 4 * N * (N * N) * h * 32          # QK^T and AV, both 2*N*N*h*32 per row
        ms = bench(lambda: ttnn.transformer.scaled_dot_product_attention(
            q, k, v, is_causal=False), n=3, warm=1)
        R["roofs"][f"sdpa_c{c}_h{h}_ms"] = round(ms, 4)
        R["roofs"][f"sdpa_c{c}_h{h}_TFLOPs"] = round(flop / (ms * 1e-3) / 1e12, 3)
        ttnn.deallocate(q); ttnn.deallocate(k); ttnn.deallocate(v)
    except Exception as e:                                                    # noqa: BLE001
        R["roofs"][f"sdpa_c{c}"] = f"ERR {type(e).__name__}: {str(e)[:110]}"

a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(R, indent=1))
print(json.dumps(R, indent=1), flush=True)
T.cleanup()
