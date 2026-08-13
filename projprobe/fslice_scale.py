#!/usr/bin/env python3
"""How does the verified 1D affine pass scale across cores, and what binds it?

fslice_pass.py verified the kernel (rel L2 4.20e-3 vs fp64, which is bf16 round-off) and measured
2788.9 ns per output tile per pass on ONE core -- 2.08x the 1338.6 ns per-output-tile reader bound S1e
measured with 130 cores active. Single-core that says compute-bound, but a single core does not contend
for L1 or DRAM, so it cannot settle what binds the kernel at scale.

A core-count sweep does. Perfectly compute-bound work keeps its per-core rate as cores are added;
bandwidth-bound work loses it. Each core takes a different source row range so no arm is flattered by
130 cores sharing one hot page.

The slices/s conversion is DERIVED and labelled as such: at box 256 a slice is 25,736 output points =
25.1 output tiles, and the design's stage 2 runs this pass three times per final output tile (2a
produces two tiles, 2b produces one). Stage 1 is not built and is not in this number.
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
SRC_W = 1024
SRC_ROWS = 256
BARRIER_EVERY = 4
A = 1.31
NB = 400                       # output tiles per core
GRIDS = ((1, 1), (4, 4), (8, 8), (13, 8), (13, 10))
TILES_PER_SLICE = 25736 / 1024.0
PASSES_PER_TILE = 3


def sel_matrices(A, nsrc):
    P = [np.zeros((nsrc, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(A * u))
        for d in range(3):
            if 0 <= k + d < nsrc:
                P[d][k + d, u] = 1.0
    return P


def build(dev, x, sel, frac, out, nx, ny, offs_bytes, mode=1):
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
    wct = ([CB_OUT, TILE_B, SRC_TILES if mode == 0 else 1]
           + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            row0 = (c * 7) % (SRC_ROWS - NROWS)
            rrt[cx][cy] = ([x.buffer_address(), NB, row0] + [int(o) for o in offs_bytes]
                           + [sel.buffer_address(), frac.buffer_address()])
            crt[cx][cy] = [NB]
            wrt[cx][cy] = [out.buffer_address(), NB, c * NB]
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
    res = {"A": A, "nb_per_core": NB, "barrier_every": BARRIER_EVERY, "grids": {}}
    try:
        rng = np.random.default_rng(21)
        src_t = torch.from_numpy(
            rng.integers(-120, 120, size=(SRC_ROWS, SRC_W)).astype(np.float32)).to(torch.bfloat16)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(src_t.reshape(1, 1, SRC_ROWS, SRC_W), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
        P = sel_matrices(A, WIN)
        selt = np.concatenate([P[d].reshape(SRC_TILES, 32, 32).reshape(-1, 32) for d in range(3)], 0)
        sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        h = np.array([math.modf(0.37 * r + 0.2)[0] for r in range(NROWS)])
        fr = np.zeros((2, 32, 32), dtype=np.float32)
        for u in range(32):
            fr[0, 0, u] = A * u - math.floor(A * u)
        for r in range(NROWS):
            fr[1, r, :] = h[r]
        frac = ttnn.from_torch(torch.from_numpy(fr.reshape(1, 1, 64, 32)).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        offs_by = (8 * np.arange(NROWS, dtype=np.int64)) * ELEM

        base = None
        for nx, ny in GRIDS:
            n = nx * ny
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n * SRC_TILES, 32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, x, sel, frac, out, nx, ny, offs_by, 1)
            ttnn.generic_op([x, sel, frac, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([x, sel, frac, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            per_core_ns = best * 1e9 / NB
            chip_tiles_s = n * NB / best
            slices_s = chip_tiles_s / (TILES_PER_SLICE * PASSES_PER_TILE)
            if base is None:
                base = per_core_ns
            res["grids"][f"{nx}x{ny}"] = {
                "ncores": n, "ns_per_output_tile_per_core": per_core_ns,
                "chip_output_tiles_per_s": chip_tiles_s, "derived_slices_per_s": slices_s,
                "per_core_slowdown_vs_1core": per_core_ns / base}
            print(f"{nx}x{ny} = {n:3d} cores: {per_core_ns:8.1f} ns/tile/core "
                  f"({per_core_ns/base:5.2f}x vs 1 core)   {chip_tiles_s/1e6:7.2f} M tiles/s chip   "
                  f"-> {slices_s/1e3:7.1f} k slices/s (DERIVED)", flush=True)
            json.dump(res, open(HERE / "fslice_scale.json", "w"), indent=1)
            ttnn.deallocate(out)

        # --- where does the time go? ----------------------------------------------------------------
        # mode 0 stops after the tilize, mode 2 after the six selection matmuls and the broadcast, and
        # mode 1 is the whole pass. Differencing them attributes the cost to the three stages without
        # a profiler, and it is the difference that matters: mode 1 minus mode 2 is exactly the six
        # SFPU ops (copy_dest_values, floor, frac, three lerps).
        nx, ny = 13, 10
        n = nx * ny
        res["attribution"] = {}
        prev = None
        for mode, label in ((0, "tilize only"), (2, "+ 6 matmuls + bcast"), (1, "+ 6 SFPU ops")):
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n * SRC_TILES, 32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, x, sel, frac, out, nx, ny, offs_by, mode)
            ttnn.generic_op([x, sel, frac, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([x, sel, frac, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            ns = best * 1e9 / NB
            delta = "" if prev is None else f"   (+{ns - prev:7.1f} ns)"
            print(f"mode {mode}  {label:22s} {ns:8.1f} ns/tile/core{delta}", flush=True)
            res["attribution"][f"mode{mode}"] = {"label": label, "ns_per_tile_per_core": ns,
                                                 "delta_ns": None if prev is None else ns - prev}
            prev = ns
            json.dump(res, open(HERE / "fslice_scale.json", "w"), indent=1)
            ttnn.deallocate(out)
    finally:
        ttnn.close_device(dev)


main()
