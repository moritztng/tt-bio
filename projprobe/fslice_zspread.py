#!/usr/bin/env python3
"""Stage 1's adjoint -- the last piece of backprojection, and the last piece of "build both".

Forward: W = sum_p mask_p * V_p, reading nplane volume tiles and writing one W tile.
Adjoint: V'_p = mask_p * W', reading one W tile and writing nplane volume tiles.

The transaction count is identical and only the direction changes, so the two should cost about the
same. Section 5 predicted exactly this shape ("spreads one value into 2 z-planes weighted by the same
mask"), and section 36 established the fact that makes it simple: each volume tile is touched by exactly
one W tile within a direction, so nothing accumulates across W tiles and every product is written once.

Correctness against fp64 is direct -- the adjoint of a weighted sum is the same weights applied to the
scattered value, so the reference is mask_p * W' per plane, with no reduction to get wrong.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent / "tt_bio" / "kernels" / "fslice"
CB_W, CB_MASK, CB_OUT = 0, 8, 16
ELEM = 2
TILE_B = 32 * 32 * ELEM
NB = 100
NPAGES = 512
PLANES = (2, 8, 28, 51)
PSI_PER_DIRECTION = 96
TILES_PER_DIRECTION = (512 // 32) ** 2 * 2
FLOOR_SLICES_S = 3.20e6
STAGE1_FWD = 755.3      # measured, section 36: 8 copies, DRAM, no reuse


def build(dev, w, m, out, nx, ny, nb, nplane, dram=True):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    rct = ([CB_W, CB_MASK, nplane, TILE_B]
           + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(m).get_compile_time_args()))
    cct = [CB_W, CB_MASK, CB_OUT, nplane]
    wct = [CB_OUT, TILE_B, nplane] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = [w.buffer_address(), m.buffer_address(), nb, (c * 7) % 64, NPAGES]
            crt[cx][cy] = [nb]
            wrt[cx][cy] = [out.buffer_address(), nb, c * nb * nplane]
            c += 1
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_zspread.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_zspread.cpp", cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(KDIR / "writer_fslice.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(CB_W, TILE_B, 4), cb(CB_MASK, TILE_B, nplane),
                           cb(CB_OUT, TILE_B, 2 * nplane)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"planes": list(PLANES), "stage1_forward_ns_per_slice": STAGE1_FWD, "arms": {}}
    try:
        rng = np.random.default_rng(91)
        wv = rng.integers(-100, 100, size=(NPAGES * 32, 32)).astype(np.float32)
        wt = torch.from_numpy(wv).to(torch.bfloat16)
        wn = wt.to(torch.float64).numpy().reshape(NPAGES, 32, 32)
        # The volume is the DESTINATION here and does not fit L1 any more than it does in the forward,
        # so it is a DRAM tensor. The W plane being read is small and L1-resident.
        wl1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        w = ttnn.from_torch(wt.reshape(1, 1, NPAGES * 32, 32), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=wl1)

        for nplane in PLANES:
            a, b = 0.43, 0.29
            mk = np.zeros((nplane, 32, 32), dtype=np.float32)
            for X in range(32):
                for Y in range(32):
                    z = a * X + b * Y
                    z0 = int(np.floor(z))
                    t = z - z0
                    if 0 <= z0 < nplane:
                        mk[z0, X, Y] = 1 - t
                    if 0 <= z0 + 1 < nplane:
                        mk[z0 + 1, X, Y] = t
            m = ttnn.from_torch(torch.from_numpy(mk.reshape(1, 1, nplane * 32, 32)).to(torch.bfloat16),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            mkn = torch.from_numpy(mk).to(torch.bfloat16).to(torch.float64).numpy()

            o1 = ttnn.from_torch(torch.zeros(1, 1, 32 * nplane, 32).to(torch.bfloat16),
                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, w, m, o1, 1, 1, 1, nplane)
            ttnn.generic_op([w, m, o1], pd)
            ttnn.synchronize_device(dev)
            g = ttnn.to_torch(o1).reshape(nplane, 32, 32).to(torch.float64).numpy()
            ref = mkn * wn[0][None, :, :]
            rel = float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-300))
            ttnn.deallocate(o1)

            nx, ny = 13, 10
            n = nx * ny
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n * nplane, 32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                                  memory_config=ttnn.MemoryConfig(
                                      ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM))
            pd = build(dev, w, m, out, nx, ny, NB, nplane)
            ttnn.generic_op([w, m, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([w, m, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            ns = best * 1e9 / NB
            per_slice = ns * TILES_PER_DIRECTION / n / PSI_PER_DIRECTION
            res["arms"][str(nplane)] = {"rel_l2": rel, "ns_per_w_tile": ns,
                                        "ns_per_slice_per_core": per_slice}
            print(f"nplane {nplane:3d}: rel L2 {rel:.3e}   {ns:9.1f} ns per W tile   "
                  f"-> {per_slice:7.1f} ns per slice per core", flush=True)
            json.dump(res, open(HERE / "fslice_zspread.json", "w"), indent=1)
            ttnn.deallocate(out)
            ttnn.deallocate(m)

        a28 = res["arms"].get("28")
        if a28:
            adj = a28["ns_per_slice_per_core"]
            print(f"\nstage 1 adjoint {adj:.1f} ns/slice/core vs forward {STAGE1_FWD:.1f} "
                  f"-> {adj/STAGE1_FWD:.2f}x", flush=True)
            res["adjoint_over_forward"] = adj / STAGE1_FWD
            json.dump(res, open(HERE / "fslice_zspread.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
