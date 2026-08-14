#!/usr/bin/env python3
"""Backprojection: the adjoint of the 1D affine pass. The second primitive the brief asks for.

Section 5 predicted the structure before any kernel existed. The forward pass is
    out[r, u] = sum_d c_d(r, u) * (srcp . P_d)[r, u]
so the adjoint scatters a slice back into the source lattice as
    src'[r, j] = sum_d (c_d(r, u) * y[r, u]) . P_d^T
Every piece transposes: the selection matrices become their own transposes, the coefficients are
unchanged, and the per-row offset moves from the reader to the writer.

Two things this measures that section 5 asserted:

1. NO ttnn.scatter, no atomics, no cross-core reduction. matmul_tiles accumulates into DST (section
   24.1), so every orientation touching a volume tile adds into the same registers under one acquire.
   The `ncontrib` sweep is that claim: if accumulation is free, cost per contribution should FALL as
   contributions are batched, because the write is paid once.

2. fp32 accumulation costs nothing on the matrix engine. S0 measured fp32 and bf16 matmul within 2%,
   and backprojection sums 10^5-10^6 contributions per voxel where bf16 would be indefensible.

Correctness is against a direct fp64 adjoint. The per-row WRITE offset is not implemented here -- the
output is the 64-wide source window before scattering -- so this measures the adjoint's arithmetic and
accumulation, not its scatter. That is named, not hidden.
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
CB_Y, CB_COEF, CB_SELT, CB_MID, CB_OUT = 0, 4, 8, 12, 16
ELEM = 2
TILE_B = 32 * 32 * ELEM
OUT_TILES = 2
NPAGES = 512
NB = 200
A = 1.31
CONTRIB = (1, 2, 4, 8)
NROWS = 32
VOL_W = 1024      # volume row pitch, elements
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


def build(dev, y, cf, st, out, nx, ny, nb, ncontrib, fp32=False, scatter=0):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    nselt = 3 * OUT_TILES
    rct = ([CB_Y, CB_COEF, CB_SELT, nselt, TILE_B]
           + list(ttnn.TensorAccessorArgs(y).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(cf).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(st).get_compile_time_args()))
    cct = [CB_Y, CB_COEF, CB_SELT, CB_MID, CB_OUT, OUT_TILES, ncontrib]
    # The 64-wide source window is OUT_TILES tiles, so each of the 32 rows carries
    # OUT_TILES * 32 elements and lands at its own offset in the volume.
    win_b = OUT_TILES * 32 * ELEM
    wct = ([CB_OUT, TILE_B, OUT_TILES, NROWS, VOL_W * ELEM, win_b, scatter]
           + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = [y.buffer_address(), cf.buffer_address(), st.buffer_address(),
                           nb * ncontrib, (c * 5) % 64, NPAGES]
            crt[cx][cy] = [nb]
            # Per-row write offsets, 16 B aligned as the alignment probe requires.
            offs = [((c * 13 + 7 * r) % 64) * 16 for r in range(NROWS)]
            wrt[cx][cy] = [out.buffer_address(), nb, (c * nb * OUT_TILES) % 4096, 4096] + offs
            c += 1
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_bproj.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_bproj.cpp", cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                       fp32_dest_acc_en=fp32)),
        mk(KDIR / "writer_bproj.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(CB_Y, TILE_B, 4), cb(CB_COEF, TILE_B, 3), cb(CB_SELT, TILE_B, nselt),
                           cb(CB_MID, TILE_B, 2 * 3 * ncontrib), cb(CB_OUT, TILE_B, 2 * OUT_TILES)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"A": A, "contrib": list(CONTRIB), "arms": {}}
    try:
        rng = np.random.default_rng(81)
        yv = rng.integers(-100, 100, size=(NPAGES * 32, 32)).astype(np.float32)
        yt = torch.from_numpy(yv).to(torch.bfloat16)
        yn = yt.to(torch.float64).numpy().reshape(NPAGES, 32, 32)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        y = ttnn.from_torch(yt.reshape(1, 1, NPAGES * 32, 32), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        h = np.array([math.modf(0.37 * r + 0.2)[0] for r in range(32)])
        cf = np.zeros((3, 32, 32), dtype=np.float32)
        for r in range(32):
            for u in range(32):
                w = h[r] + (A * u - math.floor(A * u))
                M = math.floor(w)
                wq = w - M
                cf[0, r, u] = (1 - wq) * (1 - M)
                cf[1, r, u] = (1 - wq) * M + wq * (1 - M)
                cf[2, r, u] = wq * M
        cft = ttnn.from_torch(torch.from_numpy(cf.reshape(1, 1, 96, 32)).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cfn = torch.from_numpy(cf).to(torch.bfloat16).to(torch.float64).numpy()

        # P_d^T laid out as OUT_TILES tiles per d: tile (d*OUT_TILES + k) is P_d^T[:, 32k:32k+32],
        # i.e. a 32 x 32 block mapping the slice's u index to source columns 32k..32k+31.
        P = sel_matrices(A, 32 * OUT_TILES)
        blocks = []
        for d in range(3):
            PT = P[d].T                                  # 32 (u) x 64 (j)
            for k in range(OUT_TILES):
                blocks.append(PT[:, 32 * k:32 * (k + 1)])
        selt = np.concatenate(blocks, axis=0).astype(np.float32)
        st = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                             dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        seltn = torch.from_numpy(selt).to(torch.bfloat16).to(torch.float64).numpy()

        base = None
        for fp32 in (False, True):
          tag = "fp32acc" if fp32 else "bf16acc"
          base = None
          for ncontrib in CONTRIB:
              NCHK = 1
              o1 = ttnn.from_torch(torch.zeros(1, 1, 32 * OUT_TILES, 32).to(torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
              pd = build(dev, y, cft, st, o1, 1, 1, NCHK, ncontrib, fp32)
              ttnn.generic_op([y, cft, st, o1], pd)
              ttnn.synchronize_device(dev)
              g = ttnn.to_torch(o1).reshape(OUT_TILES, 32, 32).to(torch.float64).numpy()
              # fp64 adjoint, accumulated over the same ncontrib slice tiles the reader consumed.
              ref = np.zeros((OUT_TILES, 32, 32))
              for n in range(ncontrib):
                  yy = yn[n]
                  for d in range(3):
                      wgt = cfn[d] * yy                       # 32 (r) x 32 (u)
                      for k in range(OUT_TILES):
                          ref[k] += wgt @ seltn[(d * OUT_TILES + k) * 32:(d * OUT_TILES + k + 1) * 32]
              rel = float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-300))
              ttnn.deallocate(o1)

              nx, ny = 13, 10
              n = nx * ny
              out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n * OUT_TILES, 32).to(torch.bfloat16),
                                    dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
              pd = build(dev, y, cft, st, out, nx, ny, NB, ncontrib, fp32)
              ttnn.generic_op([y, cft, st, out], pd)
              ttnn.synchronize_device(dev)
              best = float("inf")
              for _ in range(5):
                  t0 = time.perf_counter()
                  ttnn.generic_op([y, cft, st, out], pd)
                  ttnn.synchronize_device(dev)
                  best = min(best, time.perf_counter() - t0)
              ns = best * 1e9 / NB
              per_contrib = ns / ncontrib
              if base is None:
                  base = per_contrib
              res["arms"][f"{tag}/{ncontrib}"] = {"rel_l2": rel, "ns_per_volume_tile": ns,
                                            "ns_per_contribution": per_contrib,
                                            "gain_vs_1": base / per_contrib}
              if fp32 and ncontrib in (1, 8):
                  o2 = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n * OUT_TILES, 32).to(torch.bfloat16),
                                       dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                  pd2 = build(dev, y, cft, st, o2, nx, ny, NB, ncontrib, fp32, 1)
                  ttnn.generic_op([y, cft, st, o2], pd2)
                  ttnn.synchronize_device(dev)
                  b2 = float("inf")
                  for _ in range(5):
                      t0 = time.perf_counter()
                      ttnn.generic_op([y, cft, st, o2], pd2)
                      ttnn.synchronize_device(dev)
                      b2 = min(b2, time.perf_counter() - t0)
                  ns2 = b2 * 1e9 / NB
                  res["arms"][f"{tag}/{ncontrib}"]["ns_scattered_write"] = ns2
                  res["arms"][f"{tag}/{ncontrib}"]["scatter_over_bulk"] = ns2 / ns
                  print(f"      per-row scattered write: {ns2:9.1f} ns/volume-tile "
                        f"({ns2/ns:5.2f}x bulk)", flush=True)
                  ttnn.deallocate(o2)
              print(f"{tag} ncontrib {ncontrib:2d}: rel L2 {rel:.3e}   {ns:9.1f} ns/volume-tile   "
                    f"{per_contrib:8.1f} ns per contribution ({base/per_contrib:5.2f}x)", flush=True)
              json.dump(res, open(HERE / "fslice_bproj.json", "w"), indent=1)
              ttnn.deallocate(out)
    finally:
        ttnn.close_device(dev)


main()
