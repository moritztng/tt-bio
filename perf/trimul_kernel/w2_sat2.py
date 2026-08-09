#!/usr/bin/env python3
"""W2 saturation follow-ups, from what w2_saturation.py measured.

F1 -- the triangle contraction is unpack-bound, so try to parallelise over the CHANNEL axis.
Production uses MatmulMultiCoreReuseMultiCast with per_core_M = per_core_N = 1 (Kt=10 output
tiles over an 11x10 grid), so a core computes one 32x32 output tile per batch entry after
unpacking 10 in0 + 10 in1 tiles: 20 tile unpacks per output tile, no DEST reuse. Every
cores-for-reuse variant on the multicast config LOST (5x5 grid 14.24 vs 29.74 TFLOP/s),
because shrinking the grid to buy reuse costs more cores than it buys. The one arrangement
that gets both is batch-parallel: 64 channels x 100 output tiles = 6400 tiles, so a core can
own a real block of ONE channel and still fill the grid. That is MatmulMultiCoreReuse (no
multicast). It threw at program.cpp:1052 on the first attempt; print the full error and walk
the per-core block size.

F2 -- the gate is SFPU-bound, measured: plain multiply on the production shape runs at
1002.8 GB/s aggregate (90% of the 1109.8 GB/s L1 clone roof) and the same multiply with a
SIGMOID activation on operand b runs at 411.4, i.e. 2.44x slower for zero extra bytes. So the
sigmoid, not bandwidth, is what holds the gate at 37% of the copy roof. The lever is to move
it into the input projection's packer, where it can overlap the matmul's math instead of
serialising behind the eltwise unpack. That needs the fused [g|g|p|p] projection split into a
g-matmul with activation="sigmoid" and a plain p-matmul. Question this answers: is the
matmul-fused activation free, and what does the split cost?

    TT_VISIBLE_DEVICES=1 python3 perf/trimul_kernel/w2_sat2.py --n 320 --c 64
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from tt_bio.tenstorrent import CORE_GRID_MAIN, COMPUTE_GRID_MAIN, get_device

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def timeit(dev, fn, warm=3, iters=5, pipe=8):
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


def f1(dev, N, C, ckc, rows):
    kt = (N + 31) // 32
    gx, gy = COMPUTE_GRID_MAIN
    flops = C * 2 * N * N * N
    print(f"\n=== F1: batch-parallel triangle contraction, [1,{C},{N},{N}]^2, Kt={kt} ===")
    torch.manual_seed(1)
    a = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=L1)
    b = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=L1)
    prod = ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy), in0_block_w=kt,
        out_subblock_h=1, out_subblock_w=1, out_block_h=1, out_block_w=1,
        per_core_M=1, per_core_N=1, transpose_mcast=False, fused_activation=None,
        fuse_batch=False)
    ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                                    program_config=prod, dtype=ttnn.bfloat16))
    ser, pip = timeit(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=ckc,
                                               memory_config=L1, program_config=prod,
                                               dtype=ttnn.bfloat16))
    base = pip
    print(f"{'PRODUCTION (multicast, pcM=pcN=1)':52s} {pip:8.4f} ms {flops / (pip * 1e-3) / 1e12:7.2f} TF")
    rows.append(dict(part="f1", name="production", pipe_ms=round(pip, 4),
                     tflops=round(flops / (pip * 1e-3) / 1e12, 2), bit_exact=True))

    for pm, pn, sh, sw, bw in [(10, 10, 1, 1, kt), (10, 10, 2, 2, kt), (10, 10, 5, 1, kt),
                               (10, 10, 1, 5, kt), (5, 10, 1, 1, kt), (10, 5, 1, 1, kt),
                               (5, 5, 1, 1, kt), (10, 10, 2, 5, kt), (10, 10, 5, 2, kt),
                               (10, 10, 1, 2, 5), (10, 10, 2, 2, 5), (10, 10, 4, 2, 2)]:
        nm = f"MultiCoreReuse pcM={pm} pcN={pn} sub {sh}x{sw} blk_w={bw}"
        try:
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(gx, gy), in0_block_w=bw,
                out_subblock_h=sh, out_subblock_w=sw, per_core_M=pm, per_core_N=pn)
            ser, pip = timeit(dev, lambda cfg=cfg: ttnn.matmul(
                a, b, compute_kernel_config=ckc, memory_config=L1,
                program_config=cfg, dtype=ttnn.bfloat16))
            out = ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                              program_config=cfg, dtype=ttnn.bfloat16)
            eq = bool(torch.equal(ttnn.to_torch(out), ref))
            ttnn.deallocate(out)
        except Exception as e:
            msg = " ".join(str(e).split())[:230]
            print(f"{nm:52s} FAILED {msg}")
            rows.append(dict(part="f1", name=nm, error=msg))
            continue
        tf = flops / (pip * 1e-3) / 1e12
        print(f"{nm:52s} {pip:8.4f} ms {tf:7.2f} TF  {base / pip:5.3f}x  exact={eq}")
        rows.append(dict(part="f1", name=nm, pipe_ms=round(pip, 4), tflops=round(tf, 2),
                         speedup=round(base / pip, 4), bit_exact=eq))
    ttnn.deallocate(a)
    ttnn.deallocate(b)


def f2(dev, N, C, CZ, ckc, rows):
    """Is a matmul-fused sigmoid free? And what does splitting the fused projection cost?"""
    print(f"\n=== F2: sigmoid in the projection packer, [1,{N},{N},{CZ}] @ [{CZ},w] ===")
    torch.manual_seed(3)
    # Production reads x_norm_in from DRAM (the layer_norm inherits the pair tensor's
    # memory config) and writes the projection result to L1. Putting the 50 MiB operand in
    # L1 instead leaves too little L1 for the matmul's own circular buffers: every backend
    # threw "Statically allocated circular buffers ... clash with L1 buffers".
    x = ttnn.from_torch(torch.randn(1, N, N, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    w_full = ttnn.from_torch(torch.randn(CZ, 4 * C), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)
    w_half = ttnn.from_torch(torch.randn(CZ, 2 * C), layout=ttnn.TILE_LAYOUT, device=dev,
                             dtype=ttnn.bfloat16)

    def bench(nm, fn, out_w):
        # pipe=1: the projection result is 25-50 MiB in L1, and holding 8 of them at once
        # for a pipelined timing overruns the 195 MB aggregate L1 and the matmul then throws
        # "Statically allocated circular buffers ... clash with L1 buffers". Serial median.
        try:
            ser, pip = timeit(dev, fn, pipe=1)
        except Exception as e:
            msg = " ".join(str(e).split())[:200]
            print(f"{nm:52s} FAILED {msg}")
            rows.append(dict(part="f2", name=nm, error=msg))
            return None
        fl = 2 * N * N * CZ * out_w
        print(f"{nm:52s} {pip:8.4f} ms {fl / (pip * 1e-3) / 1e12:7.2f} TF")
        rows.append(dict(part="f2", name=nm, pipe_ms=round(pip, 4),
                         tflops=round(fl / (pip * 1e-3) / 1e12, 2), out_w=out_w))
        return pip

    p1 = bench("minimal_matmul  -> 4C  [PRODUCTION gp_in]",
               lambda: ttnn.experimental.minimal_matmul(x, w_full, memory_config=L1,
                                                        dtype=ttnn.bfloat16,
                                                        compute_kernel_config=ckc), 4 * C)
    p2 = bench("minimal_matmul  -> 2C  [half width, x2 = split]",
               lambda: ttnn.experimental.minimal_matmul(x, w_half, memory_config=L1,
                                                        dtype=ttnn.bfloat16,
                                                        compute_kernel_config=ckc), 2 * C)
    bench("linear          -> 4C",
          lambda: ttnn.linear(x, w_full, memory_config=L1, dtype=ttnn.bfloat16,
                              compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN), 4 * C)
    bench("linear          -> 2C",
          lambda: ttnn.linear(x, w_half, memory_config=L1, dtype=ttnn.bfloat16,
                              compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN), 2 * C)
    bench("linear+sigmoid  -> 2C  [activation in the packer]",
          lambda: ttnn.linear(x, w_half, memory_config=L1, dtype=ttnn.bfloat16,
                              compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN,
                              activation="sigmoid"), 2 * C)
    for kw in ("activation", "fused_activation"):
        try:
            f = lambda: ttnn.experimental.minimal_matmul(
                x, w_half, memory_config=L1, dtype=ttnn.bfloat16,
                compute_kernel_config=ckc, **{kw: "sigmoid"})
            bench(f"minimal_matmul+sigmoid -> 2C  [{kw}=]", f, 2 * C)
        except Exception as e:
            msg = " ".join(str(e).split())[:150]
            print(f"{'minimal_matmul+sigmoid (' + kw + ')':52s} UNSUPPORTED {msg}")
            rows.append(dict(part="f2", name=f"minimal_matmul {kw}", error=msg))
    if p1 and p2:
        print(f"\nsplit cost: 2 x {p2:.4f} = {2 * p2:.4f} ms vs one {p1:.4f} ms "
              f"-> {2 * p2 - p1:+.4f} ms per chunk")
    ttnn.deallocate(x)
    ttnn.deallocate(w_full)
    ttnn.deallocate(w_half)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--parts", default="12")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    rows = []
    if "1" in args.parts:
        f1(dev, args.n, args.c, ckc, rows)
    if "2" in args.parts:
        f2(dev, args.n, args.c, args.cz, ckc, rows)
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
