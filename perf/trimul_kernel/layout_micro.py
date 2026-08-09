#!/usr/bin/env python3
"""Two microbenchmarks behind the 298-aa trimul's non-matmul half.

M1 -- the channel move. `opsplit298.py` put the trimul's three permutes at 23% of the op,
running at 12-16% of the 528 GB/s L1->L1 clone roof measured on this card. A permute that moves
the channel axis across the tiled pair of dims is a sub-tile shuffle, so ttnn falls back to a
slow path. Every one of them factors into a tile-local `transpose(-2,-1)` plus a
`permute(0,2,1,3)` that only reorders whole tiles:

    permute(0,3,1,2)  ==  transpose(-2,-1) . permute(0,2,1,3)
    permute(0,3,2,1)  ==  transpose(-2,-1) . permute(0,2,1,3) . transpose(-2,-1)
    permute(0,2,3,1)  ==  permute(0,2,1,3) . transpose(-2,-1)

All three are pure index reorderings, so the factored form must be bit-identical; this checks
that with torch.equal rather than assuming it.

M2 -- the output projections. `ttnn.linear` gets 19 TFLOP/s on [1,N,N,c_z]@[c_z,c_z] while
`ttnn.experimental.minimal_matmul` gets 46 TFLOP/s on the identical shape inside the same
trimul. This separates the two candidate causes: the kernel, and whether the result lands in
L1 or DRAM.

    TT_VISIBLE_DEVICES=1 python3 perf/trimul_kernel/layout_micro.py --n 320 --c 64
"""

import argparse
import json
import time
from pathlib import Path

import torch

import ttnn

from tt_bio.tenstorrent import CORE_GRID_MAIN, get_device


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
    ap.add_argument("--c", type=int, default=64, help="channel chunk width")
    ap.add_argument("--cz", type=int, default=256)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    N, C, CZ = args.n, args.c, args.cz
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    L1 = ttnn.L1_MEMORY_CONFIG
    DRAM = ttnn.DRAM_MEMORY_CONFIG
    rows = []

    def bench(name, fn, nbytes, ref=None, note=""):
        ser, pip = timeit(dev, fn)
        gbs = nbytes / (pip * 1e-3) / 1e9
        eq = None
        if ref is not None:
            out = fn()
            eq = bool(torch.equal(ttnn.to_torch(out), ref))
            ttnn.deallocate(out)
        rows.append(dict(name=name, serial_ms=round(ser, 4), pipe_ms=round(pip, 4),
                         gbs_each_way=round(gbs, 1), bit_exact=eq, note=note))
        print(f"{name:44s} {pip:8.4f} ms  {gbs:7.1f} GB/s each way  "
              f"{'exact=' + str(eq) if eq is not None else ''} {note}")
        return pip

    # ---------------- M1: channel moves ----------------
    torch.manual_seed(0)
    ch = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    nb = 1 * N * N * C * 2
    print(f"\nM1 channel moves, chunk [1,{N},{N},{C}] = {nb / 2**20:.1f} MiB, L1 resident")
    bench("clone  (the L1 copy roof)", lambda: ttnn.clone(ch, memory_config=L1), nb)
    bench("transpose(-2,-1)  [tile-local]",
          lambda: ttnn.transpose(ch, -2, -1, memory_config=L1), nb)

    ref_a = ttnn.to_torch(ttnn.permute(ch, (0, 3, 1, 2), memory_config=L1))
    bench("permute(0,3,1,2)  [production]",
          lambda: ttnn.permute(ch, (0, 3, 1, 2), memory_config=L1), nb, ref_a)

    def fac_a():
        t = ttnn.transpose(ch, -2, -1, memory_config=L1)
        o = ttnn.permute(t, (0, 2, 1, 3), memory_config=L1)
        ttnn.deallocate(t)
        return o
    bench("  = transpose . permute(0,2,1,3)", fac_a, nb, ref_a)

    ref_b = ttnn.to_torch(ttnn.permute(ch, (0, 3, 2, 1), memory_config=L1))
    bench("permute(0,3,2,1)  [production]",
          lambda: ttnn.permute(ch, (0, 3, 2, 1), memory_config=L1), nb, ref_b)

    def fac_b():
        t = ttnn.transpose(ch, -2, -1, memory_config=L1)
        o = ttnn.permute(t, (0, 2, 1, 3), memory_config=L1)
        ttnn.deallocate(t)
        o2 = ttnn.transpose(o, -2, -1, memory_config=L1)
        ttnn.deallocate(o)
        return o2
    bench("  = transpose . permute(0,2,1,3) . transpose", fac_b, nb, ref_b)

    cm = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    ref_c = ttnn.to_torch(ttnn.permute(cm, (0, 2, 3, 1), memory_config=L1))
    bench("permute(0,2,3,1)  [production, out]",
          lambda: ttnn.permute(cm, (0, 2, 3, 1), memory_config=L1), nb, ref_c)

    def fac_c():
        t = ttnn.permute(cm, (0, 2, 1, 3), memory_config=L1)
        o = ttnn.transpose(t, -2, -1, memory_config=L1)
        ttnn.deallocate(t)
        return o
    bench("  = permute(0,2,1,3) . transpose", fac_c, nb, ref_c)
    bench("permute(0,2,1,3) alone  [whole-tile move]",
          lambda: ttnn.permute(cm, (0, 2, 1, 3), memory_config=L1), nb)
    ttnn.deallocate(ch)
    ttnn.deallocate(cm)

    # ---------------- M2: output projection ----------------
    print(f"\nM2 output projection [1,{N},{N},{CZ}] @ [{CZ},{CZ}], "
          f"{2 * N * N * CZ * CZ / 1e9:.2f} GFLOP")
    z = ttnn.from_torch(torch.randn(1, N, N, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    w = ttnn.from_torch(torch.randn(CZ, CZ), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16)
    fl = 2 * N * N * CZ * CZ
    pb = N * N * CZ * 2

    def tf(pip):
        return fl / (pip * 1e-3) / 1e12
    for nm, fn in [
        ("linear -> DRAM  [production]",
         lambda: ttnn.linear(z, w, memory_config=DRAM, dtype=ttnn.bfloat16,
                             compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)),
        ("linear -> L1",
         lambda: ttnn.linear(z, w, memory_config=L1, dtype=ttnn.bfloat16,
                             compute_kernel_config=ckc, core_grid=CORE_GRID_MAIN)),
        ("minimal_matmul -> DRAM",
         lambda: ttnn.experimental.minimal_matmul(z, w, memory_config=DRAM,
                                                  dtype=ttnn.bfloat16,
                                                  compute_kernel_config=ckc)),
        ("minimal_matmul -> L1",
         lambda: ttnn.experimental.minimal_matmul(z, w, memory_config=L1,
                                                  dtype=ttnn.bfloat16,
                                                  compute_kernel_config=ckc)),
    ]:
        try:
            ser, pip = timeit(dev, fn)
        except Exception as e:
            print(f"{nm:44s} FAILED: {type(e).__name__}: {str(e)[:120]}")
            rows.append(dict(name=nm, error=f"{type(e).__name__}: {str(e)[:200]}"))
            continue
        rows.append(dict(name=nm, serial_ms=round(ser, 4), pipe_ms=round(pip, 4),
                         tflops=round(tf(pip), 2), gbs_w=round(pb / (pip * 1e-3) / 1e9, 1)))
        print(f"{nm:44s} {pip:8.4f} ms  {tf(pip):6.2f} TFLOP/s  "
              f"write {pb / (pip * 1e-3) / 1e9:6.1f} GB/s")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
