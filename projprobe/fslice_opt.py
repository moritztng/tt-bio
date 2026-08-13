#!/usr/bin/env python3
"""The optimisation the attribution asked for: stop deriving M and frac(q) on the device.

Differencing the bisect modes at 130 cores put the six SFPU ops -- copy_dest_values, floor, frac and
three lerps -- at 1694.7 ns of a 2936.4 ns output tile, 58% of it, against roughly 90 ns for the six
selection matmuls plus the broadcast. The matrix engine was never the problem.

But M and frac(q) are functions of g(r) and frac(A*u), and both are known on the host. Deriving them
on the device costs a broadcast add, a copy_dest_values, a floor and a frac; reading them costs two
FPU copy_tile. Mode 4 does the latter.

Verified before it is timed, at one core against the same fp64 reference mode 1 uses, then both are
timed at 130 cores so the lever is reported as a before/after rather than an assertion.

CAVEAT, stated because it bounds the result: M and frac(q) depend on the output tile index through
g(r, t), so in the full design they are two tiles per output tile rather than two fixed tiles. This
measurement holds them fixed, so it understates the real cost by two bulk reads per output tile. The
core sweep showed the reader idle at 1.03x across a 130x fanout, so those reads should be absorbed --
but that is an expectation, not something measured here.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent / "tt_bio" / "kernels" / "fslice"
CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT = 0, 4, 8, 12, 16
NROWS, SRC_TILES = 32, 2
WIN = 32 * SRC_TILES
ELEM = 2
TILE_B = 32 * 32 * ELEM
SRC_W, SRC_ROWS = 1024, 256
BARRIER_EVERY = 4
A = 1.31
NB = 400
TILES_PER_SLICE = 25736 / 1024.0
PASSES_PER_TILE = 3
FLOOR_SLICES_S = 3.20e6          # section 3.2, box 256, measured 420.2 GB/s roof


def sel_matrices(A, nsrc):
    P = [np.zeros((nsrc, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(A * u))
        for d in range(3):
            if 0 <= k + d < nsrc:
                P[d][k + d, u] = 1.0
    return P


def build(dev, x, sel, frac, out, nx, ny, offs_bytes, mode, nb):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    rct = ([CB_SRC, WIN * ELEM, NROWS, SRC_W * ELEM, SRC_TILES, BARRIER_EVERY, mode,
            CB_SEL, CB_FRAC, 3 * SRC_TILES, TILE_B]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(frac).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, SRC_TILES, mode]
    wct = [CB_OUT, TILE_B, 1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            row0 = (c * 7) % (SRC_ROWS - NROWS)
            rrt[cx][cy] = ([x.buffer_address(), nb, row0] + [int(o) for o in offs_bytes]
                           + [sel.buffer_address(), frac.buffer_address()])
            crt[cx][cy] = [nb]
            wrt[cx][cy] = [out.buffer_address(), nb, c * nb]
            c += 1
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_fslice.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_fslice.cpp", cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(KDIR / "writer_fslice.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(CB_SRC, TILE_B, 4 * BARRIER_EVERY * SRC_TILES),
                           cb(CB_TIL, TILE_B, 2 * SRC_TILES), cb(CB_SEL, TILE_B, 3 * SRC_TILES),
                           cb(CB_FRAC, TILE_B, 2), cb(CB_OUT, TILE_B, 4)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"A": A, "nb_per_core": NB, "arms": {}}
    try:
        rng = np.random.default_rng(21)
        src_t = torch.from_numpy(
            rng.integers(-120, 120, size=(SRC_ROWS, SRC_W)).astype(np.float32)).to(torch.bfloat16)
        srcn = src_t.to(torch.float64).numpy()
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(src_t.reshape(1, 1, SRC_ROWS, SRC_W), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
        P = sel_matrices(A, WIN)
        selt = np.concatenate([P[d].reshape(SRC_TILES, 32, 32).reshape(-1, 32) for d in range(3)], 0)
        sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        h = np.array([math.modf(0.37 * r + 0.2)[0] for r in range(NROWS)])
        offs_el = 8 * np.arange(NROWS, dtype=np.int64)
        offs_by = offs_el * ELEM

        # mode 1 operands: frac(A*u) in row 0 of tile 0, g(r) replicated in tile 1.
        f1 = np.zeros((2, 32, 32), dtype=np.float32)
        for u in range(32):
            f1[0, 0, u] = A * u - math.floor(A * u)
        for r in range(NROWS):
            f1[1, r, :] = h[r]
        frac1 = ttnn.from_torch(torch.from_numpy(f1.reshape(1, 1, 64, 32)).to(torch.bfloat16),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        # mode 4 operands: frac(q) in tile 0, M in tile 1 -- both computed here instead of on device.
        f4 = np.zeros((2, 32, 32), dtype=np.float32)
        for r in range(NROWS):
            for u in range(32):
                w = h[r] + (A * u - math.floor(A * u))
                f4[0, r, u] = w - math.floor(w)
                f4[1, r, u] = math.floor(w)
        frac4 = ttnn.from_torch(torch.from_numpy(f4.reshape(1, 1, 64, 32)).to(torch.bfloat16),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        ref = np.zeros((32, 32))
        for r in range(NROWS):
            for u in range(32):
                q = A * u + h[r]
                j = math.floor(q)
                f = q - j
                b0 = offs_el[r] + j
                ref[r, u] = (1 - f) * srcn[r + 0, b0] + f * srcn[r + 0, b0 + 1]

        # --- verify mode 4 at one core BEFORE timing it ---------------------------------------------
        out1 = ttnn.from_torch(torch.zeros(1, 1, 32, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)
        for mode, fr in ((1, frac1), (4, frac4)):
            pd = build(dev, x, sel, fr, out1, 1, 1, offs_by, mode, 1)
            ttnn.generic_op([x, sel, fr, out1], pd)
            ttnn.synchronize_device(dev)
            g = ttnn.to_torch(out1).reshape(32, 32).to(torch.float64).numpy()
            rel = float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-300))
            res["arms"][f"mode{mode}_rel_l2"] = rel
            print(f"mode {mode} vs fp64 (1 core): rel L2 {rel:.4e}", flush=True)
        ttnn.deallocate(out1)

        # --- time both at 130 cores -----------------------------------------------------------------
        nx, ny = 13, 10
        n = nx * ny
        timings = {}
        for mode, fr in ((1, frac1), (4, frac4)):
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n, 32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, x, sel, fr, out, nx, ny, offs_by, mode, NB)
            ttnn.generic_op([x, sel, fr, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([x, sel, fr, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            ns = best * 1e9 / NB
            tiles_s = n * NB / best
            slices_s = tiles_s / (TILES_PER_SLICE * PASSES_PER_TILE)
            timings[mode] = ns
            res["arms"][f"mode{mode}_130core"] = {
                "ns_per_output_tile_per_core": ns, "chip_output_tiles_per_s": tiles_s,
                "derived_slices_per_s": slices_s,
                "pct_of_floor": 100.0 * slices_s / FLOOR_SLICES_S}
            print(f"mode {mode} @130 cores: {ns:8.1f} ns/tile/core   {tiles_s/1e6:6.2f} M tiles/s   "
                  f"-> {slices_s/1e3:7.1f} k slices/s ({100*slices_s/FLOOR_SLICES_S:4.1f}% of floor)",
                  flush=True)
            json.dump(res, open(HERE / "fslice_opt.json", "w"), indent=1)
            ttnn.deallocate(out)
        print(f"\nlever: host-precomputed M and frac(q) bought "
              f"{timings[1]/timings[4]:.3f}x ({timings[1]-timings[4]:.1f} ns per output tile)",
              flush=True)
        res["lever_speedup"] = timings[1] / timings[4]
        json.dump(res, open(HERE / "fslice_opt.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
