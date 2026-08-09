#!/usr/bin/env python3
"""Which half of the channel-major transform is the cost, and what does the fused
kernel have to beat?

W4 probe 2. Probe 1 (`chanmajor_probe.py`) established that `permute(0,3,1,2)` runs at
15.8-24.7% of the measured L1->L1 copy roof on identical bytes, and that
`transpose(-2,-1) + permute(0,2,1,3)` is bit-exact with it and 1.15x faster. It did NOT
split that pair, so we do not know which half is off the roof. That decides the kernel
boundary:

  * `transpose(-2,-1)` maps input tile (j,c) -> (c,j) ONE-FOR-ONE: whole tile in, whole
    tile out. A fused kernel can absorb it for free alongside the gate.
  * `permute(0,2,1,3)` swaps a batch dim with the tile-ROW dim, so element (i,c,j) moves
    from intra-tile row c%32 to intra-tile row i%32. That is a sub-tile shuffle and it is
    the only part of the cube transpose that cannot be done with whole-tile moves.

So: if the pair's cost sits in `transpose(-2,-1)`, the cheap fused kernel (gate + tile
transpose, all whole tiles) captures it. If it sits in `permute(0,2,1,3)`, the kernel has
to do a row-granular shuffle and that is a different, riskier design.

This probe also times every op the fused input-side kernel is meant to replace, on the
exact production shapes at N=320 (chunk width 64, 4 pairs, L1), so milestone 1 has a
baseline measured in one place under one timing discipline.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/permute_halves.py --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

from tt_bio.tenstorrent import get_device

L1 = ttnn.L1_MEMORY_CONFIG


def timeit(dev, fn, warm=3, iters=9):
    for _ in range(warm):
        r = fn()
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
        if isinstance(r, ttnn.Tensor):
            ttnn.deallocate(r)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64, help="production chunk width")
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = get_device()
    N, C, CZ = args.n, args.c, args.c_z
    torch.manual_seed(0)
    rows = []

    def add(name, fn, mb=None, note=""):
        ms = timeit(dev, fn)
        gbs = (mb / ms * 1000 / 1e3) if mb else None
        rows.append(dict(part=name, ms=round(ms, 4), moved_mb=round(mb, 2) if mb else None,
                         eff_gbs=round(gbs, 1) if gbs else None, note=note))
        extra = f" {mb:7.1f}MB {gbs:7.1f}GB/s" if mb else ""
        print(f"  {name:46s} {ms:8.4f} ms{extra}  {note}", flush=True)
        return ms

    tin = torch.randn(1, N, N, C)
    x = ttnn.from_torch(tin, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                        memory_config=L1)
    rw = 2 * (N * N * C * 2) / 1e6

    print(f"\n=== the transform, [1,{N},{N},{C}] bf16 = {N*N*C*2/1e6:.1f} MB, L1->L1 ===",
          flush=True)
    add("clone (copy roof, no reorder)", lambda: ttnn.clone(x, memory_config=L1), rw)
    add("permute(0,3,1,2)  [production 'a']",
        lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=L1), rw)
    add("permute(0,3,2,1)  [production 'b']",
        lambda: ttnn.permute(x, (0, 3, 2, 1), memory_config=L1), rw)

    # half 1: tile-local, whole-tile-in/whole-tile-out
    xt = ttnn.transpose(x, -2, -1, memory_config=L1)   # [1,N,C,N]
    add("half1: transpose(-2,-1)  [1,N,N,C]->[1,N,C,N]",
        lambda: ttnn.transpose(x, -2, -1, memory_config=L1), rw)
    # half 2: batch-dim <-> tile-row-dim, sub-tile shuffle
    add("half2: permute(0,2,1,3)  [1,N,C,N]->[1,C,N,N]",
        lambda: ttnn.permute(xt, (0, 2, 1, 3), memory_config=L1), rw)

    def pair():
        t2 = ttnn.transpose(x, -2, -1, memory_config=L1)
        r = ttnn.permute(t2, (0, 2, 1, 3), memory_config=L1)
        ttnn.deallocate(t2)
        return r

    ref_a = tin.to(torch.bfloat16).float().permute(0, 3, 1, 2)
    got = ttnn.to_torch(pair()).float()
    add("half1+half2 (pair)", pair, rw, note=f"exact_vs_permute0312={torch.equal(got, ref_a)}")

    # is transpose(1,3) a single-op route to the 'b' operand?
    try:
        ref_b = tin.to(torch.bfloat16).float().permute(0, 3, 2, 1)
        g = ttnn.to_torch(ttnn.transpose(x, 1, 3, memory_config=L1)).float()
        add("transpose(1,3) [single-op 'b']",
            lambda: ttnn.transpose(x, 1, 3, memory_config=L1), rw,
            note=f"exact_vs_permute0321={torch.equal(g, ref_b)}")
    except Exception as e:
        print(f"  transpose(1,3) UNSUPPORTED: {type(e).__name__}: {str(e).splitlines()[0][:110]}",
              flush=True)
    ttnn.deallocate(xt)

    # ---- the chain the fused kernel replaces, production shapes -------------------
    print(f"\n=== chain to replace, per channel chunk (gp_in_fused [1,{N},{N},{4*C}]) ===",
          flush=True)
    gp = ttnn.from_torch(torch.randn(1, N, N, 4 * C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    gp_rw = (N * N * 4 * C * 2) / 1e6
    add("chunk(gp_in_fused, 4, dim=-1)", lambda: ttnn.chunk(gp, chunks=4, dim=-1),
        gp_rw * 2)
    ga = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    pa = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)

    def gate():
        return ttnn.multiply(pa, ga, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                             memory_config=L1)

    add("multiply(p, g, SIGMOID(b))  [one gate]", gate, 3 * (N * N * C * 2) / 1e6)
    mask = ttnn.from_torch(torch.ones(1, N, N, 1), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16, memory_config=L1)
    try:
        add("multiply(a, mask_u)  [mask]",
            lambda: ttnn.multiply(pa, mask, memory_config=L1), rw)
    except Exception as e:
        print(f"  mask multiply: {type(e).__name__}: {str(e).splitlines()[0][:110]}", flush=True)
    add("reallocate", lambda: ttnn.reallocate(ttnn.clone(pa, memory_config=L1)), None)

    print("\n" + json.dumps(rows, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(n=N, c=C, c_z=CZ, rows=rows), indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
