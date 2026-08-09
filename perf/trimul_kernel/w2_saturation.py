#!/usr/bin/env python3
"""W2 saturation pass: what actually binds the trimul's two biggest non-permute ops.

WARROOM 2.7 says an op at neither roof is a defect with a mechanism, not a result. Two
trimul ops are exactly there at 298 aa (N=320, c_z=256, chunk width 64):

  matmul.triangle   23.3 TFLOP/s = 17% of the 137.1 TFLOP/s HiFi4 compute roof
  gate.sigmoid_mul  298 GB/s aggregate = 28% of the 1056 GB/s aggregate L1 copy roof

Part E -- the gate. The production gate is ONE op already: multiply_(p, g, b_activations
=[SIGMOID]), 2 reads + 1 in-place write. The clone roof is a 1-read-1-write single-operand
copy, which the datamover can run without the unpacker. So the question is whether 28% is
the SFPU sigmoid, the two-operand unpack path, or real bandwidth. Ladder, same shape, same
L1 residency:
    clone            1r1w, no math      -> the copy roof
    sigmoid          1r1w, SFPU         -> isolates the transcendental
    add_             2r1w, no SFPU      -> the two-operand eltwise roof
    multiply_        2r1w, no SFPU      -> ditto, FPU mul instead of add
    multiply_+SIGMOID 2r1w, SFPU        -> production
If add_ ~= multiply_+SIGMOID the sigmoid is free and the limiter is the 2-operand path.
If multiply_+SIGMOID is materially slower, the SFPU is the limiter and the lever is real.

Part M -- the triangle matmul. Production config at N=320 is per_core_M = per_core_N = 1
(seq_len_tiles=10 over an 11x10 grid), i.e. every core computes ONE 32x32 output tile per
batch entry, reading in0_block_w=10 in0 tiles and 10 in1 tiles to do it: 20 tile unpacks
per output tile, no DEST-register reuse across neighbouring output tiles. The alternative
is to trade cores for reuse -- a smaller grid with per_core_M/N > 1 and a real out_subblock
-- which does NOT change the K accumulation order, so any winner here should still be
bit-exact against production. Sweep it and find out.

Part K -- the shape-achievable compute roof by K, so that "% of the compute roof" has a
denominator that some backend actually reaches at this K (W8: no trunk op reaches 137.1).

    TT_VISIBLE_DEVICES=1 python3 perf/trimul_kernel/w2_saturation.py --n 320 --c 64
"""

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

import ttnn

from tt_bio.tenstorrent import CORE_GRID_MAIN, COMPUTE_GRID_MAIN, get_device

L1 = ttnn.L1_MEMORY_CONFIG
DRAM = ttnn.DRAM_MEMORY_CONFIG


def timeit(dev, fn, warm=3, iters=5, pipe=10, free=True):
    """Median serial ms and pipelined ms. Synced on both sides of every timed region."""
    for _ in range(warm):
        r = fn()
        if free:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ser = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ser.append((time.perf_counter() - t0) * 1e3)
        if free:
            ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    outs = [fn() for _ in range(pipe)]
    ttnn.synchronize_device(dev)
    pip = (time.perf_counter() - t0) * 1e3 / pipe
    if free:
        for o in outs:
            ttnn.deallocate(o)
    return sorted(ser)[len(ser) // 2], pip


def part_eltwise(dev, N, C, ckc, rows):
    """The gate ladder. Every op on the identical [1,N,N,C] L1-resident shape."""
    nb = N * N * C * 2                       # one tensor, bytes
    print(f"\n=== Part E: the gate ladder, [1,{N},{N},{C}] = {nb / 2**20:.1f} MiB, L1 ===")
    print(f"{'op':46s} {'pipe ms':>9s} {'r+w MB':>8s} {'GB/s agg':>9s} {'note'}")

    def mk():
        torch.manual_seed(0)
        return ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.bfloat16, memory_config=L1)

    def bench(name, fn, traffic_mb, free=True, note=""):
        try:
            ser, pip = timeit(dev, fn, free=free)
        except Exception as e:
            print(f"{name:46s} FAILED {type(e).__name__}: {str(e)[:90]}")
            rows.append(dict(part="eltwise", name=name, error=f"{type(e).__name__}: {str(e)[:200]}"))
            return None
        gbs = traffic_mb * 1e6 / (pip * 1e-3) / 1e9
        rows.append(dict(part="eltwise", name=name, pipe_ms=round(pip, 4),
                         serial_ms=round(ser, 4), traffic_mb=round(traffic_mb, 1),
                         gbs_aggregate=round(gbs, 1), note=note))
        print(f"{name:46s} {pip:9.4f} {traffic_mb:8.1f} {gbs:9.1f}  {note}")
        return pip

    mb = nb / 1e6
    a, b = mk(), mk()
    bench("clone            1r1w  [the L1 copy roof]",
          lambda: ttnn.clone(a, memory_config=L1), 2 * mb)
    bench("sigmoid          1r1w  SFPU",
          lambda: ttnn.sigmoid(b, memory_config=L1), 2 * mb)
    bench("exp              1r1w  SFPU",
          lambda: ttnn.exp(b, memory_config=L1), 2 * mb)
    bench("add              2r1w  FPU, out-of-place",
          lambda: ttnn.add(a, b, memory_config=L1), 3 * mb)
    bench("multiply         2r1w  FPU, out-of-place",
          lambda: ttnn.multiply(a, b, memory_config=L1), 3 * mb)
    bench("multiply+SIGMOID 2r1w  out-of-place [production math]",
          lambda: ttnn.multiply(a, b, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                                memory_config=L1), 3 * mb)

    # In-place forms: these mutate their first operand, so each call needs a fresh victim.
    # Timing them the same way would measure the from_torch too, so instead time N calls
    # against N pre-made victims.
    def inplace(name, op_kwargs, traffic_mb):
        vic = [mk() for _ in range(14)]
        try:
            for v in vic[:4]:
                ttnn.multiply_(v, b, **op_kwargs)
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            for v in vic[4:]:
                ttnn.multiply_(v, b, **op_kwargs)
            ttnn.synchronize_device(dev)
            pip = (time.perf_counter() - t0) * 1e3 / len(vic[4:])
        except Exception as e:
            print(f"{name:46s} FAILED {type(e).__name__}: {str(e)[:90]}")
            rows.append(dict(part="eltwise", name=name, error=f"{type(e).__name__}: {str(e)[:200]}"))
            for v in vic:
                ttnn.deallocate(v)
            return None
        for v in vic:
            ttnn.deallocate(v)
        gbs = traffic_mb * 1e6 / (pip * 1e-3) / 1e9
        rows.append(dict(part="eltwise", name=name, pipe_ms=round(pip, 4),
                         traffic_mb=round(traffic_mb, 1), gbs_aggregate=round(gbs, 1)))
        print(f"{name:46s} {pip:9.4f} {traffic_mb:8.1f} {gbs:9.1f}")
        return pip

    inplace("multiply_        2r1w  in-place, no activation", {}, 3 * mb)
    inplace("multiply_+SIGMOID 2r1w in-place  [PRODUCTION]",
            dict(input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]), 3 * mb)
    ttnn.deallocate(a)
    ttnn.deallocate(b)


def tri_cfg(seq_len_tiles, gx, gy, in0_block_w=None, sub_h=1, sub_w=1):
    per_core_M = -(-seq_len_tiles // gy)
    per_core_N = -(-seq_len_tiles // gx)
    if in0_block_w is None:
        in0_block_w = max(d for d in range(min(10, seq_len_tiles), 0, -1)
                          if seq_len_tiles % d == 0)
    return ttnn.MatmulMultiCoreReuseMultiCastProgramConfig(
        compute_with_storage_grid_size=(gx, gy),
        in0_block_w=in0_block_w,
        out_subblock_h=sub_h,
        out_subblock_w=sub_w,
        out_block_h=per_core_M,
        out_block_w=per_core_N,
        per_core_M=per_core_M,
        per_core_N=per_core_N,
        transpose_mcast=False,
        fused_activation=None,
        fuse_batch=False,
    )


def part_matmul(dev, N, C, ckc, rows):
    """The triangle contraction: production config vs cores-for-reuse alternatives."""
    kt = (N + 31) // 32
    gx, gy = COMPUTE_GRID_MAIN
    flops = C * 2 * N * N * N
    print(f"\n=== Part M: triangle matmul [1,{C},{N},{N}] @ [1,{C},{N},{N}], "
          f"Kt={kt}, grid {gx}x{gy}, {flops / 1e9:.2f} GFLOP/call ===")
    torch.manual_seed(1)
    ta = torch.randn(1, C, N, N)
    tb = torch.randn(1, C, N, N)
    a = ttnn.from_torch(ta, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                        memory_config=L1)
    b = ttnn.from_torch(tb, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                        memory_config=L1)

    prod_cfg = tri_cfg(kt, gx, gy)
    ref = ttnn.to_torch(ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                                    program_config=prod_cfg, dtype=ttnn.bfloat16))

    print(f"{'config':56s} {'pipe ms':>9s} {'TFLOP/s':>9s} {'exact':>6s}")

    def bench(name, fn):
        try:
            ser, pip = timeit(dev, fn)
            out = fn()
            eq = bool(torch.equal(ttnn.to_torch(out), ref))
            ttnn.deallocate(out)
        except Exception as e:
            print(f"{name:56s} FAILED {type(e).__name__}: {str(e)[:70]}")
            rows.append(dict(part="matmul", name=name,
                             error=f"{type(e).__name__}: {str(e)[:200]}"))
            return None
        tf = flops / (pip * 1e-3) / 1e12
        rows.append(dict(part="matmul", name=name, pipe_ms=round(pip, 4),
                         serial_ms=round(ser, 4), tflops=round(tf, 2), bit_exact=eq))
        print(f"{name:56s} {pip:9.4f} {tf:9.2f} {str(eq):>6s}")
        return pip

    bench(f"PRODUCTION pc grid {gx}x{gy} pcM=pcN=1 sub 1x1 blk_w={kt}",
          lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                              program_config=prod_cfg, dtype=ttnn.bfloat16))
    bench("no program_config, core_grid=CORE_GRID_MAIN",
          lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                              core_grid=CORE_GRID_MAIN, dtype=ttnn.bfloat16))
    bench("no program_config, ttnn default placement",
          lambda: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                              dtype=ttnn.bfloat16))

    # Trade cores for DEST reuse. per_core_M = ceil(kt/gy), per_core_N = ceil(kt/gx),
    # so a smaller grid gives every core a real block of output tiles and lets the
    # subblock be > 1x1. K order is untouched, so these should stay bit-exact.
    for (ggx, ggy, sh, sw) in [(5, 5, 2, 2), (5, 5, 1, 2), (5, 5, 2, 1),
                               (5, 2, 5, 2), (2, 5, 2, 5), (4, 4, 3, 3),
                               (3, 3, 4, 4), (10, 5, 2, 1), (5, 10, 1, 2),
                               (10, 10, 1, 1), (2, 2, 5, 5)]:
        pm, pn = -(-kt // ggy), -(-kt // ggx)
        if pm % sh or pn % sw:
            continue
        bench(f"grid {ggx}x{ggy} pcM={pm} pcN={pn} sub {sh}x{sw} blk_w={kt}",
              lambda cfg=tri_cfg(kt, ggx, ggy, sub_h=sh, sub_w=sw):
              ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                          program_config=cfg, dtype=ttnn.bfloat16))

    # in0_block_w below full K: changes the K accumulation order, so not bit-exact.
    for bw in [1, 2, 5]:
        bench(f"PRODUCTION grid, blk_w={bw} (K order changes)",
              lambda cfg=tri_cfg(kt, gx, gy, in0_block_w=bw):
              ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                          program_config=cfg, dtype=ttnn.bfloat16))

    # Batch-parallel reuse config: distributes batch*M over cores instead of mcasting.
    for (ggx, ggy) in [(11, 10), (8, 8)]:
        try:
            cfg = ttnn.MatmulMultiCoreReuseProgramConfig(
                compute_with_storage_grid_size=(ggx, ggy),
                in0_block_w=kt, out_subblock_h=1, out_subblock_w=1,
                per_core_M=kt, per_core_N=kt)
        except Exception as e:
            print(f"MatmulMultiCoreReuse {ggx}x{ggy} cfg build FAILED: {str(e)[:80]}")
            continue
        bench(f"MultiCoreReuse (batch-parallel) grid {ggx}x{ggy}",
              lambda cfg=cfg: ttnn.matmul(a, b, compute_kernel_config=ckc, memory_config=L1,
                                          program_config=cfg, dtype=ttnn.bfloat16))

    # DRAM-resident operands, production config: is the contraction sensitive to where
    # the operands live at all? If not, it is not bandwidth-bound.
    ad = ttnn.to_memory_config(a, DRAM)
    bd = ttnn.to_memory_config(b, DRAM)
    bench("PRODUCTION config, operands + result in DRAM",
          lambda: ttnn.matmul(ad, bd, compute_kernel_config=ckc, memory_config=DRAM,
                              program_config=prod_cfg, dtype=ttnn.bfloat16))
    ttnn.deallocate(ad)
    ttnn.deallocate(bd)
    ttnn.deallocate(a)
    ttnn.deallocate(b)


def part_k(dev, N, ckc, rows):
    """Shape-achievable compute roof vs K on this card, at the trimul's M,N."""
    print(f"\n=== Part K: achievable TFLOP/s vs K, [1,{N},{N},K] @ [K,K] ===")
    print(f"{'shape':40s} {'backend':18s} {'pipe ms':>9s} {'TFLOP/s':>9s}")
    for K in [64, 128, 256, 512, 1024]:
        torch.manual_seed(2)
        try:
            z = ttnn.from_torch(torch.randn(1, N, N, K), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=DRAM)
            w = ttnn.from_torch(torch.randn(K, K), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16)
        except Exception as e:
            print(f"K={K} alloc FAILED: {str(e)[:80]}")
            continue
        fl = 2 * N * N * K * K
        for nm, fn in [
            ("linear", lambda: ttnn.linear(z, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                                           compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)),
            ("minimal_matmul", lambda: ttnn.experimental.minimal_matmul(
                z, w, memory_config=DRAM, dtype=ttnn.bfloat16, compute_kernel_config=ckc)),
        ]:
            try:
                ser, pip = timeit(dev, fn, warm=2, iters=3, pipe=6)
            except Exception as e:
                print(f"{f'[1,{N},{N},{K}]@[{K},{K}]':40s} {nm:18s} FAILED "
                      f"{type(e).__name__}: {str(e)[:60]}")
                continue
            tf = fl / (pip * 1e-3) / 1e12
            rows.append(dict(part="k_sweep", K=K, backend=nm, pipe_ms=round(pip, 4),
                             tflops=round(tf, 2)))
            print(f"{f'[1,{N},{N},{K}]@[{K},{K}]':40s} {nm:18s} {pip:9.4f} {tf:9.2f}")
        ttnn.deallocate(z)
        ttnn.deallocate(w)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--parts", default="EMK")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    rows = []
    try:
        if "E" in args.parts:
            part_eltwise(dev, args.n, args.c, ckc, rows)
        if "M" in args.parts:
            part_matmul(dev, args.n, args.c, ckc, rows)
        if "K" in args.parts:
            part_k(dev, args.n, ckc, rows)
    except Exception:
        traceback.print_exc()
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
