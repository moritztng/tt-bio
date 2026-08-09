#!/usr/bin/env python3
"""Does the trimul's channel-move permute run at 100 GB/s because it is core-starved?

`layout_micro.py` showed the trimul's three permutes at 79-104 GB/s against a 540 GB/s L1->L1
clone roof on the same 12.5 MiB chunk, and showed the factored forms are no better -- even
`permute(0,2,1,3)`, a pure whole-tile reorder with no sub-tile shuffling, costs the same. So
the slow part is not the shuffle granularity.

Hypothesis: ttnn's permute parallelizes over the channel axis, which at the production chunk
width C=64 is two tiles. Prediction: hold the moved bytes fixed and vary C -- if the op is
core-starved in C, time per byte falls sharply as C grows and flattens once C/32 exceeds the
grid. If instead it is bandwidth-limited, GB/s is flat in C.

Falsifier: flat GB/s across C=32..256 kills it.

    TT_VISIBLE_DEVICES=1 python3 perf/trimul_kernel/permute_width.py --n 320
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from tt_bio.tenstorrent import get_device


def timeit(dev, fn, warm=4, iters=7, pipe=10):
    for _ in range(warm):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ser = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ser.append((time.perf_counter() - t0) * 1e3)
        ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(pipe)]
    ttnn.synchronize_device(dev)
    pip = (time.perf_counter() - t0) * 1e3 / pipe
    for o in outs:
        ttnn.deallocate(o)
    return sorted(ser)[len(ser) // 2], pip


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    N = args.n
    dev = get_device()
    rows = []
    print(f"N={N}. chunk [1,{N},{N},C] -> permute(0,3,1,2) -> [1,C,{N},{N}], L1 resident.")
    print(f"{'C':>5s} {'MiB':>7s} {'clone ms':>9s} {'clone GB/s':>11s} "
          f"{'perm ms':>9s} {'perm GB/s':>10s} {'ratio':>6s} {'ms/chan':>8s}")
    for C in (32, 64, 128, 256):
        nb = N * N * C * 2
        torch.manual_seed(0)
        try:
            x = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=ttnn.L1_MEMORY_CONFIG)
            _, cl = timeit(dev, lambda: ttnn.clone(x, memory_config=ttnn.L1_MEMORY_CONFIG))
            _, pm = timeit(dev, lambda: ttnn.permute(x, (0, 3, 1, 2),
                                                     memory_config=ttnn.L1_MEMORY_CONFIG))
            ttnn.deallocate(x)
        except Exception as e:
            print(f"{C:5d} FAILED: {type(e).__name__}: {str(e)[:100]}")
            rows.append(dict(c=C, error=f"{type(e).__name__}: {str(e)[:200]}"))
            continue
        cg, pg = nb / (cl * 1e-3) / 1e9, nb / (pm * 1e-3) / 1e9
        rows.append(dict(c=C, mib=round(nb / 2**20, 2), clone_ms=round(cl, 4),
                         clone_gbs=round(cg, 1), permute_ms=round(pm, 4),
                         permute_gbs=round(pg, 1), ratio=round(cg / pg, 2),
                         ms_per_channel=round(pm / C, 6)))
        print(f"{C:5d} {nb / 2**20:7.1f} {cl:9.4f} {cg:11.1f} {pm:9.4f} {pg:10.1f} "
              f"{cg / pg:6.2f} {pm / C * 1e3:8.4f}")

    # same question in DRAM: the >352-aa path lives there
    print("\nDRAM resident, same shapes:")
    for C in (64, 256):
        nb = N * N * C * 2
        torch.manual_seed(0)
        x = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        _, cl = timeit(dev, lambda: ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG))
        _, pm = timeit(dev, lambda: ttnn.permute(x, (0, 3, 1, 2),
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG))
        ttnn.deallocate(x)
        cg, pg = nb / (cl * 1e-3) / 1e9, nb / (pm * 1e-3) / 1e9
        rows.append(dict(c=C, mem="dram", clone_ms=round(cl, 4), clone_gbs=round(cg, 1),
                         permute_ms=round(pm, 4), permute_gbs=round(pg, 1),
                         ratio=round(cg / pg, 2)))
        print(f"{C:5d} {nb / 2**20:7.1f} {cl:9.4f} {cg:11.1f} {pm:9.4f} {pg:10.1f} "
              f"{cg / pg:6.2f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
