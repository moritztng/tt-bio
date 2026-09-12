#!/usr/bin/env python3
"""Paired A/B of ttnn.layer_norm with and without Welford, at Boltz-2's pair-track shape.

tt-metal's default layernorm compute kernel makes five full-width pack passes over every row
(layernorm.cpp:186, :214, :269, :305, :332); the Welford variant
(layernorm_large_tensor_welford.cpp, selected at layernorm_device_operation.cpp:215) exists to
collapse the mean/variance passes. LayerNorm is 10.1 % of the Boltz-2 Pairformer block
(b2z-kernel-cycle-census) and tt-bio passes no program config to ttnn.layer_norm anywhere, so it
has never run this path.

Interleaved A/B in one process with an A/A floor, per CONTEXT §5. Welford changes the reduction
order, so this is NOT bit-exact: a win here has to clear the cdk2x2_298 structural control before
it goes anywhere near main.

    python3 welford_ab.py --device-id 0 [--seq 512 --dim 128 --iters 20]
"""
import argparse
import statistics
import time

import torch
import ttnn


def _bench(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device-id", type=int, default=0)
    ap.add_argument("--seq", type=int, default=512, help="pair-track token axis")
    ap.add_argument("--dim", type=int, default=128, help="c_z")
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()

    dev = ttnn.open_device(device_id=a.device_id)
    try:
        shape = (1, a.seq, a.seq, a.dim)
        def dev_tensor(t):
            return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

        x = dev_tensor(torch.randn(*shape, dtype=torch.bfloat16))
        w = dev_tensor(torch.ones(a.dim, dtype=torch.bfloat16))
        b = dev_tensor(torch.zeros(a.dim, dtype=torch.bfloat16))

        grid = dev.compute_with_storage_grid_size()
        crs = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                ttnn.CoreCoord(grid.x - 1, grid.y - 1))})
        recip = ttnn.create_layer_norm_reciprocals(dev, crs, a.dim)

        stock = ttnn.LayerNormDefaultProgramConfig()
        welford = ttnn.LayerNormDefaultProgramConfig(use_welford=True)

        def run(cfg, r):
            kw = {"weight": w, "bias": b, "epsilon": 1e-5, "program_config": cfg}
            if r is not None:
                kw["recip_tensor"] = r
            y = ttnn.layer_norm(x, **kw)
            ttnn.synchronize_device(dev)
            ttnn.deallocate(y)

        _bench(lambda: run(stock, None), 0, warmup=3)      # compile both program variants first
        _bench(lambda: run(welford, recip), 0, warmup=3)

        legs = {"stock": [], "welford": [], "stock_aa": []}
        for _ in range(a.iters):                      # interleaved, one process
            legs["stock"] += _bench(lambda: run(stock, None), 1, warmup=0)
            legs["welford"] += _bench(lambda: run(welford, recip), 1, warmup=0)
            legs["stock_aa"] += _bench(lambda: run(stock, None), 1, warmup=0)

        med = {k: statistics.median(v) for k, v in legs.items()}
        aa = abs(med["stock_aa"] - med["stock"]) / med["stock"] * 100
        print(f"shape          {shape}")
        print(f"stock          {med['stock']:.4f} ms")
        print(f"welford        {med['welford']:.4f} ms")
        print(f"A/A floor      {aa:.3f} %")
        print(f"RATIO          {med['stock'] / med['welford']:.4f}x")
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
