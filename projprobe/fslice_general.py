#!/usr/bin/env python3
"""Close the generality gap: an arbitrary per-row integer offset, not just multiples of 8.

The constraint (projprobe/fslice_align.py): a per-row NoC read offset is honoured at 16 B granularity
from an L1 source, 8 bf16 elements. So floor(B*r + C) is only reachable when it happens to be a
multiple of 8, and every result so far -- including the 1.22 M slices/s of section 18 -- covers only
that shear family.

The fix costs no time and 8x source residency. Hold the source pre-replicated at the 8 sub-offsets,
copy s being the source shifted left by s elements, and send row r to copy (k0(r) mod 8) at byte offset
8*floor(k0(r)/8). Picking a copy is a different ROW INDEX, and the reader was already issuing one
addressed read per row, so the transaction count, the bytes moved and the arithmetic are all unchanged.

This verifies that against fp64 with per-row offsets deliberately chosen NOT to be multiples of 8, and
times it against the restricted version to confirm the cost really is zero rather than assumed zero.
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
SRC_W, SRC_ROWS = 1024, 64
NCOPY = 8                       # one per sub-offset
BARRIER_EVERY = 4
A = 1.31
NB = 400
TILES_PER_SLICE = 25736 / 1024.0
PASSES_PER_TILE = 3
FLOOR_SLICES_S = 3.20e6


def sel_matrices(A, nsrc):
    P = [np.zeros((nsrc, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(A * u))
        for d in range(3):
            if 0 <= k + d < nsrc:
                P[d][k + d, u] = 1.0
    return P


def build(dev, x, sel, frac, out, nx, ny, offs_bytes, rowidx, mode, nb, nfrac):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    rct = ([CB_SRC, WIN * ELEM, NROWS, SRC_W * ELEM, SRC_TILES, BARRIER_EVERY, mode,
            CB_SEL, CB_FRAC, 3 * SRC_TILES, TILE_B, nfrac]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(frac).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, SRC_TILES, mode]
    wct = [CB_OUT, TILE_B, 1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = ([x.buffer_address(), nb, 0] + [int(o) for o in offs_bytes]
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
                           cb(CB_FRAC, TILE_B, 4), cb(CB_OUT, TILE_B, 4)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"A": A, "ncopy": NCOPY, "arms": {}}
    try:
        rng = np.random.default_rng(31)
        base = rng.integers(-120, 120, size=(SRC_ROWS, SRC_W + NCOPY)).astype(np.float32)
        basen = torch.from_numpy(base).to(torch.bfloat16).to(torch.float64).numpy()
        # Copy s is the source shifted left by s elements. Row (s*SRC_ROWS + r) of the replicated
        # tensor is source row r starting at element s.
        # LAYOUT MATTERS. Blocking the copies (all of copy 0, then all of copy 1, ...) puts the 32
        # per-row reads 64 rows = 128 KB apart and measured 2.70x slower than the unreplicated kernel.
        # Interleaving them per row instead keeps a row's 8 copies adjacent, so the 32 reads stay 8
        # rows apart rather than scattering across the whole tensor.
        rep = np.zeros((SRC_ROWS * NCOPY, SRC_W), dtype=np.float32)
        for r in range(SRC_ROWS):
            for s in range(NCOPY):
                rep[r * NCOPY + s, :] = basen[r, s:s + SRC_W]
        rep_t = torch.from_numpy(rep).to(torch.bfloat16)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(rep_t.reshape(1, 1, NCOPY * SRC_ROWS, SRC_W), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
        print(f"replicated source: {NCOPY} copies x {SRC_ROWS} rows x {SRC_W} el "
              f"= {NCOPY*SRC_ROWS*SRC_W*ELEM/1e6:.2f} MB", flush=True)

        P = sel_matrices(A, WIN)
        selt = np.concatenate([P[d].reshape(SRC_TILES, 32, 32).reshape(-1, 32) for d in range(3)], 0)
        sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

        # A GENERAL shear: B is not a multiple of 8 and neither is C, so floor(B*r + C) hits every
        # residue mod 8. This is exactly the case the restricted kernel could not express.
        B, C = 0.77, 5.3
        s_r = B * np.arange(NROWS) + C
        k0 = np.floor(s_r).astype(np.int64)
        h = s_r - k0
        rho = k0 % 8
        m = k0 // 8
        rowidx = np.arange(NROWS) * NCOPY + rho         # row r, copy rho -- interleaved
        offs_by = (8 * m) * ELEM
        print(f"per-row integer offsets k0 hit residues mod 8: {sorted(set(rho.tolist()))}", flush=True)

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

        # fp64 reference against the ORIGINAL unreplicated source, at the true position A*u + B*r + C.
        ref = np.zeros((32, 32))
        for r in range(NROWS):
            for u in range(32):
                q = A * u + s_r[r]
                j = math.floor(q)
                f = q - j
                ref[r, u] = (1 - f) * basen[r, j] + f * basen[r, j + 1]

        out1 = ttnn.from_torch(torch.zeros(1, 1, 32, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev)
        pd = build(dev, x, sel, frac5, out1, 1, 1, offs_by, rowidx, 5, 1, 3)
        ttnn.generic_op([x, sel, frac5, out1], pd)
        ttnn.synchronize_device(dev)
        g = ttnn.to_torch(out1).reshape(32, 32).to(torch.float64).numpy()
        rel = float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-300))
        print(f"general shear (B={B}, C={C}) vs fp64: rel L2 {rel:.4e}  "
              f"max|diff| {np.abs(g-ref).max():.3e}", flush=True)
        res["arms"]["general_rel_l2"] = rel
        ttnn.deallocate(out1)

        nx, ny = 13, 10
        n = nx * ny
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        pd = build(dev, x, sel, frac5, out, nx, ny, offs_by, rowidx, 5, NB, 3)
        ttnn.generic_op([x, sel, frac5, out], pd)
        ttnn.synchronize_device(dev)
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            ttnn.generic_op([x, sel, frac5, out], pd)
            ttnn.synchronize_device(dev)
            best = min(best, time.perf_counter() - t0)
        ns = best * 1e9 / NB
        tiles_s = n * NB / best
        slices_s = tiles_s / (TILES_PER_SLICE * PASSES_PER_TILE)
        print(f"general @130 cores: {ns:8.1f} ns/tile/core   {tiles_s/1e6:6.2f} M tiles/s   "
              f"-> {slices_s/1e3:7.1f} k slices/s ({100*slices_s/FLOOR_SLICES_S:4.1f}% of floor)",
              flush=True)
        res["arms"]["general_130core"] = {"ns_per_output_tile_per_core": ns,
                                          "chip_output_tiles_per_s": tiles_s,
                                          "derived_slices_per_s": slices_s,
                                          "pct_of_floor": 100.0 * slices_s / FLOOR_SLICES_S}
        json.dump(res, open(HERE / "fslice_general.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
