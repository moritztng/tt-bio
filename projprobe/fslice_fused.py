#!/usr/bin/env python3
"""Both stage-2 passes fused, intermediate never leaving L1. Section 3.2's Floor B.

Every slices/s figure so far is DERIVED by assuming three passes of the measured single pass compose at
no extra cost. Section 22.2 priced the two materialised routes at 1.55x (pass 1 scatters 32 rows into a
real 2D intermediate) and 2-3x (pass 2 reads a block-per-page one), and named fusion as the route that
pays neither.

Fused, pass 1 runs twice to build the 64-wide window pass 2 needs, the two intermediate tiles are
transposed in place because the second pass resamples the other axis, and pass 2 reads them straight
out of a circular buffer. Pass 2 therefore issues NO per-row reads: the reader supplies two assemblies
per output tile where the derived composition assumed three, and no intermediate reaches DRAM.

The comparison that matters is mode 13 against 3x mode 12, since 3x mode 12 is exactly what the derived
numbers assumed. Both passes here use the same selection matrices and coefficients, which is not a real
projection geometry -- a real one has different A per pass -- but it is structurally identical in cost
and is verified against an fp64 model of the same composed operation, so it measures the composition
honestly.
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
CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, CB_MID, CB_INT, CB_INT2 = 0, 4, 8, 12, 16, 20, 24, 28
NROWS, SRC_TILES = 32, 2
WIN = 32 * SRC_TILES
ELEM = 2
TILE_B = 32 * 32 * ELEM
SRC_W, SRC_ROWS = 1024, 64
BARRIER_EVERY = 4
A = 1.31
NB = 400
TILES_PER_SLICE = 25736 / 1024.0
FLOOR_SLICES_S = 3.20e6


def sel_matrices(A, nsrc):
    P = [np.zeros((nsrc, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(A * u))
        for d in range(3):
            if 0 <= k + d < nsrc:
                P[d][k + d, u] = 1.0
    return P


def build(dev, x, sel, frac, out, nx, ny, offs_bytes, rowidx, mode, nb):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    # mode 13 consumes two assemblies per output tile, so the reader is asked for twice as many.
    nread = nb * 2 if mode == 13 else nb
    rct = ([CB_SRC, WIN * ELEM, NROWS, SRC_W * ELEM, SRC_TILES, BARRIER_EVERY, mode,
            CB_SEL, CB_FRAC, 3 * SRC_TILES, TILE_B, 3]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(frac).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, SRC_TILES, mode, CB_MID, CB_INT, CB_INT2]
    wct = [CB_OUT, TILE_B, 1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = ([x.buffer_address(), nread, 0] + [int(o) for o in offs_bytes]
                           + [sel.buffer_address(), frac.buffer_address()]
                           + [int(i) for i in rowidx])
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
                           cb(CB_FRAC, TILE_B, 4), cb(CB_OUT, TILE_B, 4), cb(CB_MID, TILE_B, 2),
                           cb(CB_INT, TILE_B, 4), cb(CB_INT2, TILE_B, 4)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"A": A, "arms": {}}
    try:
        rng = np.random.default_rng(51)
        base = rng.integers(-120, 120, size=(SRC_ROWS, SRC_W)).astype(np.float32)
        basen = torch.from_numpy(base).to(torch.bfloat16).to(torch.float64).numpy()
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(torch.from_numpy(base).to(torch.bfloat16).reshape(1, 1, SRC_ROWS, SRC_W),
                            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                            memory_config=mc)
        P = sel_matrices(A, WIN)
        selt = np.concatenate([P[d].reshape(SRC_TILES, 32, 32).reshape(-1, 32) for d in range(3)], 0)
        sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        h = np.array([math.modf(0.37 * r + 0.2)[0] for r in range(NROWS)])
        offs_el = 8 * np.arange(NROWS, dtype=np.int64)
        offs_by = offs_el * ELEM
        rowidx = np.arange(NROWS, dtype=np.int64)
        f5 = np.zeros((3, 32, 32), dtype=np.float32)
        for r in range(NROWS):
            for u in range(32):
                w = h[r] + (A * u - math.floor(A * u))
                M = math.floor(w)
                wq = w - M
                f5[0, r, u] = (1 - wq) * (1 - M)
                f5[1, r, u] = (1 - wq) * M + wq * (1 - M)
                f5[2, r, u] = wq * M
        frac5 = ttnn.from_torch(torch.from_numpy(f5.reshape(1, 1, 96, 32)).to(torch.bfloat16),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        # fp64 model of the fused composition. Pass 1 twice gives a 64-wide intermediate; the transpose
        # makes pass 2's axis the fast one; pass 2 resamples that with the same weights.
        def pass1(win):
            o = np.zeros((32, 32))
            for r in range(NROWS):
                for u in range(32):
                    q = A * u + h[r]
                    j = math.floor(q)
                    f = q - j
                    o[r, u] = (1 - f) * win[r, j] + f * win[r, j + 1]
            return o
        w0 = np.stack([basen[r, offs_el[r]:offs_el[r] + WIN] for r in range(NROWS)])
        # Pass 1s two tiles STACK vertically, so the intermediate is 64 v-rows x 32 u-columns.
        # Transposing gives 32 u-rows x 64 v-columns, which is the 64-wide window pass 2 resamples.
        I = np.concatenate([pass1(w0), pass1(w0)], axis=0)      # 64 x 32, halves identical in this test
        IT = I.T                                                # 32 x 64
        ref = np.zeros((32, 32))
        for r in range(32):
            for u in range(32):
                q = A * u + h[r]
                j = math.floor(q)
                f = q - j
                ref[r, u] = (1 - f) * IT[r, j] + f * IT[r, j + 1]

        out1 = ttnn.from_torch(torch.zeros(1, 1, 32, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)
        pd = build(dev, x, sel, frac5, out1, 1, 1, offs_by, rowidx, 13, 1)
        ttnn.generic_op([x, sel, frac5, out1], pd)
        ttnn.synchronize_device(dev)
        g = ttnn.to_torch(out1).reshape(32, 32).to(torch.float64).numpy()
        rel = float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-300))
        print(f"mode 13 fused two-pass vs fp64: rel L2 {rel:.4e}  max|diff| {np.abs(g-ref).max():.3e}",
              flush=True)
        res["arms"]["fused_rel_l2"] = rel
        ttnn.deallocate(out1)

        nx, ny = 13, 10
        n = nx * ny
        t = {}
        for mode in (12, 13):
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n, 32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, x, sel, frac5, out, nx, ny, offs_by, rowidx, mode, NB)
            ttnn.generic_op([x, sel, frac5, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([x, sel, frac5, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            ns = best * 1e9 / NB
            t[mode] = ns
            lbl = "single pass" if mode == 12 else "FUSED two-pass"
            print(f"mode {mode} {lbl:15s}: {ns:8.1f} ns per FINAL output tile per core", flush=True)
            res["arms"][f"mode{mode}_ns"] = ns
            ttnn.deallocate(out)
            json.dump(res, open(HERE / "fslice_fused.json", "w"), indent=1)

        # Per FINAL output tile per core, a target of R slices/s allows 5.18/R microseconds
        # (25,736 output points = 25.1 output tiles per slice at box 256, over 130 cores).
        derived = 3 * t[12]
        sl = lambda ns: 5.18e3 / ns * 1e3          # ns per tile -> k slices/s
        print(f"\nderived composition (3 x single pass): {derived:8.1f} ns -> {sl(derived):8.1f} k "
              f"slices/s ({100 * sl(derived) * 1e3 / FLOOR_SLICES_S:4.1f}% of floor)", flush=True)
        print(f"MEASURED fused composition:            {t[13]:8.1f} ns -> {sl(t[13]):8.1f} k "
              f"slices/s ({100 * sl(t[13]) * 1e3 / FLOOR_SLICES_S:4.1f}% of floor)", flush=True)
        print(f"fusion buys {derived / t[13]:.3f}x over the derived composition", flush=True)
        res["derived_3x_ns"] = derived
        res["fused_ns"] = t[13]
        res["fusion_speedup"] = derived / t[13]
        res["derived_k_slices_per_s"] = sl(derived)
        res["fused_k_slices_per_s"] = sl(t[13])
        res["fused_pct_of_floor"] = 100 * sl(t[13]) * 1e3 / FLOOR_SLICES_S
        json.dump(res, open(HERE / "fslice_fused.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
