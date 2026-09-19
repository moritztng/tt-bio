#!/usr/bin/env python3
"""Price every op the 12-channel pair-bias axis touches, at 12 logical channels and at 32.

The site join (`site_census.py`) puts 18.49 % of the step in ONE child program: a `FillPad` that
`ttnn.reshape` dispatches when `_linear`'s backward flattens the pair-bias gradient, whose last
axis is 12 logical channels inside a 32-wide tile. Two more entries on the same axis add 4.9 %.

Every one of those tensors is ALREADY 32 wide in memory -- 12 is the logical extent, the tile is
the storage -- so widening the host weight to 32 output columns moves no bytes and computes no new
tiles. That is a prediction, and this prices it: each op the axis passes through, at 12 and at 32,
warm, with a synchronize inside the clock.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device
from tt_bio import abodybuilder3_ops as ops


def timed(dev, fn, *, reps=5):
    fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        out.append(time.perf_counter() - t0)
        del r
    return statistics.median(out) * 1e3


def up(dev, *shape):
    return ttnn.from_torch(torch.randn(*shape), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--c-z", type=int, default=132)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    b, n, cz = args.batch, args.tokens, args.c_z
    flat = b * n * n

    dev = get_device()
    try:
        z = up(dev, b, n, n, cz)
        rows = []
        for h in (12, 32):
            g = up(dev, b, n, n, h)                       # the pair-bias gradient
            g2 = up(dev, flat, h)
            x2 = up(dev, flat, cz)
            w = up(dev, cz, h)
            bias = up(dev, h)
            hp = up(dev, b, h, n, n)
            cfg = ops.kernel_config()
            rows.append((h, [
                ("reshape 4d->2d  [b,n,n,H] -> [b*n*n,H]", timed(dev, lambda: ttnn.reshape(g, [flat, h]), reps=args.reps)),
                ("reshape 4d->3d  [b,n,n,H] -> [b*n,n,H]", timed(dev, lambda: ttnn.reshape(g, [b * n, n, h]), reps=args.reps)),
                ("reshape 4d->4d  [b,n,n,H] -> [1,1,b*n*n,H]", timed(dev, lambda: ttnn.reshape(g, [1, 1, flat, h]), reps=args.reps)),
                ("linear   z @ w   [b,n,n,cz] @ [cz,H]", timed(dev, lambda: ttnn.linear(z, w, bias=bias, compute_kernel_config=cfg), reps=args.reps)),
                ("permute  [b,n,n,H] -> [b,H,n,n]", timed(dev, lambda: ttnn.permute(g, [0, 3, 1, 2]), reps=args.reps)),
                ("matmul   dW [b*n*n,cz]^T @ [b*n*n,H]", timed(dev, lambda: ttnn.matmul(x2, g2, transpose_a=True, compute_kernel_config=cfg), reps=args.reps)),
                ("sum      dbias [b*n*n,H] -> [1,H]", timed(dev, lambda: ttnn.sum(g2, dim=0, keepdim=True), reps=args.reps)),
                ("matmul   dx [b*n*n,H] @ [cz,H]^T", timed(dev, lambda: ttnn.matmul(g2, w, transpose_b=True, compute_kernel_config=cfg), reps=args.reps)),
                ("permute  back [b,H,n,n] -> [b,n,n,H]", timed(dev, lambda: ttnn.permute(hp, [0, 2, 3, 1]), reps=args.reps)),
            ]))
            del g, g2, x2, w, bias, hp

        print(f"\nb={b} n={n} c_z={cz}, fp32, warm, median of {args.reps}, ms")
        print(f"{'op':<46} {'H=12':>9} {'H=32':>9} {'delta':>9}")
        for (name, t12), (_n2, t32) in zip(rows[0][1], rows[1][1]):
            print(f"{name:<46} {t12:>9.3f} {t32:>9.3f} {t32 - t12:>+9.3f}")
        tot12 = sum(t for _n, t in rows[0][1])
        tot32 = sum(t for _n, t in rows[1][1])
        print(f"{'sum of the rows above':<46} {tot12:>9.3f} {tot32:>9.3f} {tot32 - tot12:>+9.3f}")

        # The slice the padded forward needs, and the concat its backward needs.
        big = up(dev, b, 32, n, n)
        small = up(dev, b, 12, n, n)
        zeros = ttnn.zeros([b, 20, n, n], layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
        print(f"\nwhat padding the weight ADDS, on the untiled head axis")
        print(f"{'slice  [b,32,n,n] -> [b,12,n,n] (dim 1)':<46} "
              f"{timed(dev, lambda: ttnn.slice(big, [0, 0, 0, 0], [b, 12, n, n], [1, 1, 1, 1]), reps=args.reps):>9.3f}")
        print(f"{'concat [b,12,n,n] + [b,20,n,n] (dim 1)':<46} "
              f"{timed(dev, lambda: ttnn.concat([small, zeros], dim=1), reps=args.reps):>9.3f}")
        return 0
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    raise SystemExit(main())
