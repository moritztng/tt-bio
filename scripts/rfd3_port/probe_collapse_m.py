"""LEVER 1: does folding a batched linear's batch dim into M make it fewer/bigger matmuls?

RFD3's per-design tensors are [D, I, I, C] (pair) or [1, D, I, C] (token). ttnn runs
`matmul(in0=[D,I,I,C], in1=[C,N])` as D*I independent M=I matmuls, so device time scales
exactly 8x from D=1 to D=8 and the core count does not move (p13). The Protenix-v2
multiplicity path (tt_bio/protenix.py, 60062d4de) got 3.15-3.19x at M=4 by carrying the
batch dim through as a leading dim -- but the win there came from *sharing* the
sample-invariant conditioning, not from collapsing M.

This probe isolates the collapse itself: same data, same weight, run as
  (a) the current batched form,
  (b) reshape -> [1, 1, batch*M, K] -> one matmul -> reshape back,
in both the tile-aligned (I=256) and the real ragged (I=250, padded to 256) case, so the
reshape cost is separated from the matmul win.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import ttnn  # noqa: E402
from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device  # noqa: E402

REPEAT = 5


def ckc():
    dev = get_device()
    cls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
           else ttnn.types.BlackholeComputeKernelConfig)
    return cls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
               fp32_dest_acc_en=True, packer_l1_acc=True)


def tt(x):
    return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=get_device(),
                           dtype=ttnn.bfloat16)


def bench(fn):
    dev = get_device()
    out = fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPEAT):
        out = fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / REPEAT * 1e3, out


CASES = [
    # (I, C, N) -- I=250 is the real ragged token count, I=256 the tile-aligned one
    (250, 128, 512), (256, 128, 512),
    (250, 512, 128), (256, 512, 128),
    (250, 128, 256), (256, 128, 256),
]


def main():
    dev = get_device()
    print(f"grid {dev.compute_with_storage_grid_size().x}x{dev.compute_with_storage_grid_size().y}")
    kw = dict(compute_kernel_config=ckc(), dtype=ttnn.bfloat16)
    for I, C, N in CASES:
        for D in (1, 8):
            torch.manual_seed(0)
            a = torch.randn(1, I, I, C).repeat(D, 1, 1, 1)
            b = torch.randn(C, N)
            at, bt = tt(a), tt(b)
            n1 = I * I * N

            def cur():
                return ttnn.matmul(at, bt, **kw)

            def collapsed():
                x = ttnn.reshape(at, (1, 1, D * I * I, C))
                y = ttnn.matmul(x, bt, **kw)
                return ttnn.reshape(y, (D, I, I, N))

            def collapsed_grid():
                x = ttnn.reshape(at, (1, 1, D * I * I, C))
                y = ttnn.matmul(x, bt, core_grid=CORE_GRID_MAIN, **kw)
                return ttnn.reshape(y, (D, I, I, N))

            tc, oc = bench(cur)
            ref = ttnn.to_torch(oc).float().flatten()[:n1]
            print(f"\n[{I}x{I}x{C} @ {C}x{N}] D={D}  M={I} batch={D*I} -> collapsed M={D*I*I}")
            print(f"    {'batched (current)':<26s} {tc:8.2f} ms   1.00x")
            for label, fn in (("collapsed M", collapsed), ("collapsed M + core_grid", collapsed_grid)):
                try:
                    t, o = bench(fn)
                    d = (ttnn.to_torch(o).float().flatten()[:n1] - ref).abs().max().item()
                    print(f"    {label:<26s} {t:8.2f} ms  {tc / t:5.2f}x  "
                          f"maxabs-vs-batched {d:.3e}{'  EXACT' if d == 0.0 else ''}")
                except Exception as e:
                    print(f"    {label:<26s} FAILED {type(e).__name__}: {str(e)[:60]}")
            ttnn.deallocate(at)
            ttnn.deallocate(bt)


if __name__ == "__main__":
    main()
