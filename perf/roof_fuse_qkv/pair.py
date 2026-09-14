#!/usr/bin/env python3
"""The two halves of the ranked pair, timed against each other on one card in one process.

The fold deletes the qkv projection program and hands its work to the SDPA's reader. So what the
fold can win is bounded by what the projection costs, and what it must pay is the SDPA getting
slower on the bytes it takes on. Both measured here, interleaved, warm, synchronized, with the
projection repeated as its own A/A arm.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ttnn
import torch

from tt_bio import tenstorrent as TT
from tt_bio import triatt_qkv as TQ
from tt_bio import triatt_sdpa as TS

MB = 1e6
S = int(os.environ.get("ROOF_S", "512"))
H, DH, C = 4, 32, 128
REPS = int(os.environ.get("ROOF_REPS", "7"))


def main():
    dev = TT.get_device()
    grid = tuple(TT.COMPUTE_GRID_MAIN)
    print(f"host={os.uname().nodename} grid={grid} arch={dev.arch()} S={S}")

    torch.manual_seed(0)
    mk = lambda shape: ttnn.from_torch(  # noqa: E731
        torch.randn(*shape, dtype=torch.float32) * 0.1, layout=ttnn.TILE_LAYOUT,
        device=dev, dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    x = mk([S, S, C])
    w = mk([C, 3 * H * DH])
    bias = mk([1, H, S, S])
    scale = DH ** -0.5
    ckc = TT._compute_kernel_config() if hasattr(TT, "_compute_kernel_config") else None
    if ckc is None:
        ckc = ttnn.WormholeComputeKernelConfig(
            math_fidelity=ttnn.MathFidelity.HiFi2, math_approx_mode=True,
            fp32_dest_acc_en=False, packer_l1_acc=False)
    cfg = TT._qkv_mm_config(x, w)
    print("qkv mm config:", "None" if cfg is None else type(cfg).__name__)

    qkv = TQ.qkv_heads(x, w, ckc, H, DH, ttnn.bfloat16, cfg)
    print("head-major qkv served:", qkv is not None, "rejects:", dict(TQ.REJECTS))
    if qkv is None:
        raise SystemExit("head-major qkv declined; nothing to time against the trace's program")
    q, k, v = qkv
    ttnn.synchronize_device(dev)

    def run_mm():
        o = TQ.qkv_heads(x, w, ckc, H, DH, ttnn.bfloat16, cfg)
        return list(o)

    def run_sdpa():
        o = TS.sdpa(q, k, v, bias, scale, 512 if S >= 512 else S, 256 if S >= 256 else S)
        assert o is not None
        return [o]

    arms = [("qkv_mm", run_mm), ("sdpa", run_sdpa), ("AA:qkv_mm", run_mm)]
    for _n, f in arms:                      # warm twice: JIT then program cache
        for _ in range(2):
            for t in f():
                ttnn.deallocate(t)
    ttnn.synchronize_device(dev)

    samples = {n: [] for n, _ in arms}
    for r in range(REPS):
        for n, f in arms:
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            outs = f()
            ttnn.synchronize_device(dev)
            samples[n].append((time.perf_counter() - t0) * 1e3)
            for t in outs:
                ttnn.deallocate(t)
        print(f"  rep {r + 1}/{REPS}: " + "  ".join(
            f"{n}={samples[n][-1]:.3f}" for n, _ in arms), flush=True)

    elem = 2
    b_mm = S * S * C * elem + 3 * S * H * S * DH * elem
    cores = (grid[0] * grid[1] // H) * H
    b_sdpa = (S * H * S * DH * elem            # q
              + 2 * S * H * S * DH * elem      # k, v, read once (q_pf = 1)
              + cores * S * S * elem           # persistent mask, once per core
              + S * H * S * DH * elem)         # out
    med = {n: statistics.median(v) for n, v in samples.items()}
    print()
    for n, mb in (("qkv_mm", b_mm / MB), ("sdpa", b_sdpa / MB), ("AA:qkv_mm", b_mm / MB)):
        print(f"  {n:10s} {med[n]:7.3f} ms   {mb:7.1f} MB   {mb / med[n]:6.1f} GB/s"
              .replace("GB/s", "MB/ms"))
    aa = abs(med["AA:qkv_mm"] - med["qkv_mm"])
    print(f"  A/A floor {aa:.3f} ms ({aa / med['qkv_mm'] * 100:.2f} %)")
    print(f"  pair total {med['qkv_mm'] + med['sdpa']:.3f} ms")

    out = dict(host=os.uname().nodename, grid=list(grid), S=S, reps=REPS,
               median_ms=med, samples=samples, bytes_mb=dict(qkv_mm=b_mm / MB, sdpa=b_sdpa / MB),
               aa_ms=aa)
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"pair_{S}.json")
    json.dump(out, open(dst, "w"), indent=1)
    print("wrote", dst)


if __name__ == "__main__":
    main()
