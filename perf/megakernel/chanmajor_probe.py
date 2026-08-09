#!/usr/bin/env python3
"""Is the trimul's channel-major transform intrinsic, or an op-choice defect?

W4 (fused Pairformer megakernel), probe 1. `perf/trunk_layout/layer_split.py --n 320`
puts the two trimuls at 36.0% of a 40.30 ms Pairformer block at the 298 aa shape
(N=320, c_z=256), and inside one trimul the triangle matmul is 0.572 ms of 7.24 ms
while the layout-only ops (chunk + three permutes + concat, per channel chunk) are
~2.8 ms. Those ops do zero arithmetic.

The contraction O[c,i,j] = sum_k a[c,i,k] b[c,j,k] needs channel-major operands, so
SOME transposition is intrinsic. This probe asks two separate questions:

  Q1  How fast can [1,H,H,C] -> [1,C,H,H] be done at all? Routes: the production
      single permute, a reshape+2D-transpose+reshape, and a tile-local transpose
      followed by an outer-dim-only permute. Roof reference: ttnn.clone of the same
      tensor (a pure copy, same bytes, no reordering).

  Q2  Does the transform have to be paid on 2*hidden channels (a and b, as today) or
      on c_z channels (x_norm_in, once, before the projection)? Projecting in
      channel-major means W^T[4*hidden, c_z] @ xn^T[c_z, H*H], whose output is already
      channel-major. At protenix-v2 that is 256 channels transposed instead of 512.

Every timed region is bracketed by ttnn.synchronize_device. Correctness of each route
is checked against torch.permute on the same input, bit-exactly.

    TT_VISIBLE_DEVICES=3 python3 perf/megakernel/chanmajor_probe.py --n 320
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

import ttnn

from tt_bio.tenstorrent import get_device


def timeit(dev, fn, warm=3, iters=9):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c-z", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True,
    )
    N, CZ = args.n, args.c_z
    HID = CZ  # protenix-v2 trimul hidden == c_z
    torch.manual_seed(0)
    rows = []

    def add(name, fn, mb=None, note=""):
        ms = timeit(dev, fn)
        gbs = (mb / 1024 / ms * 1000) if mb else None
        rows.append(dict(part=name, ms=round(ms, 4),
                         moved_mb=round(mb, 2) if mb else None,
                         eff_gbs=round(gbs, 1) if gbs else None, note=note))
        extra = f"   {mb:7.1f} MB r+w   {gbs:7.1f} GB/s" if mb else ""
        print(f"  {name:44s} {ms:7.3f} ms{extra}  {note}", flush=True)

    # ---- Q1: the channel-major transform, on one production chunk width -------------
    for C in (64, CZ):
        t = torch.randn(1, N, N, C)
        x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                            memory_config=ttnn.L1_MEMORY_CONFIG)
        bytes_rw = 2 * (N * N * C * 2) / 1e6  # read + write, MB
        print(f"\n[C={C}] tensor {N}x{N}x{C} bf16 = {N*N*C*2/1e6:.1f} MB", flush=True)
        add(f"C={C} clone (copy roof, no reorder)",
            lambda x=x: ttnn.clone(x, memory_config=ttnn.L1_MEMORY_CONFIG), bytes_rw)
        add(f"C={C} permute(0,3,1,2)  [production]",
            lambda x=x: ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.L1_MEMORY_CONFIG),
            bytes_rw)

        def route_r2(x=x, C=C):
            r = ttnn.reshape(x, (1, N * N, C))
            t2 = ttnn.transpose(r, -2, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
            return ttnn.reshape(t2, (1, C, N, N))

        def route_r3(x=x, C=C):
            t2 = ttnn.transpose(x, -2, -1, memory_config=ttnn.L1_MEMORY_CONFIG)
            return ttnn.permute(t2, (0, 2, 1, 3), memory_config=ttnn.L1_MEMORY_CONFIG)

        for nm, fn in (("reshape+transpose2D+reshape", route_r2),
                       ("transpose(-2,-1)+permute(0,2,1,3)", route_r3)):
            try:
                got = ttnn.to_torch(fn()).float()
                # The device holds bf16; compare against the bf16 round-trip of the
                # same input, or every pure-movement route reads as inexact.
                ref = t.to(torch.bfloat16).float().permute(0, 3, 1, 2)
                ok = torch.equal(got, ref)
                add(f"C={C} {nm}", fn, bytes_rw, note=f"exact={ok}")
            except Exception as e:  # a route ttnn refuses is a finding, not a crash
                print(f"  C={C} {nm:38s} UNSUPPORTED: {type(e).__name__}: "
                      f"{str(e).splitlines()[0][:120]}", flush=True)
                rows.append(dict(part=f"C={C} {nm}", ms=None, note="unsupported"))
        ttnn.deallocate(x)

    # ---- Q2: projection in channel-major vs channel-last ----------------------------
    print(f"\n[projection] x_norm {N}x{N}x{CZ} -> 4*hidden={4*HID} channels", flush=True)
    xn_t = torch.randn(1, N, N, CZ)
    xn = ttnn.from_torch(xn_t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    w_chunk = ttnn.from_torch(torch.randn(CZ, 4 * 64), layout=ttnn.TILE_LAYOUT,
                              device=dev, dtype=ttnn.bfloat16)
    add("proj: minimal_matmul x1 chunk (C=64)",
        lambda: ttnn.experimental.minimal_matmul(
            xn, w_chunk, memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
            compute_kernel_config=ckc))

    # channel-major: W^T [4*hidden, c_z] @ xn^T [c_z, H*H]
    xn2 = ttnn.reshape(xn, (1, N * N, CZ))
    xnT = ttnn.transpose(xn2, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    add("proj: transpose x_norm -> [c_z, H*H]",
        lambda: ttnn.transpose(xn2, -2, -1, memory_config=ttnn.DRAM_MEMORY_CONFIG),
        2 * (N * N * CZ * 2) / 1e6)
    wT = ttnn.from_torch(torch.randn(4 * HID, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16)
    try:
        add("proj: matmul [4h,c_z]x[c_z,H*H] (all channels)",
            lambda: ttnn.matmul(wT, xnT, compute_kernel_config=ckc,
                                memory_config=ttnn.DRAM_MEMORY_CONFIG,
                                dtype=ttnn.bfloat16))
    except Exception as e:
        print(f"  channel-major projection UNSUPPORTED: {type(e).__name__}: "
              f"{str(e).splitlines()[0][:160]}", flush=True)
        rows.append(dict(part="proj channel-major", ms=None, note="unsupported"))

    print("\nrows:", json.dumps(rows, indent=2), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(n=N, c_z=CZ, rows=rows), indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
