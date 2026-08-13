#!/usr/bin/env python3
"""Stage 2 of Fourier-slice projection: the 1D affine resampling pass, verified against fp64.

What the alignment probe established, and why this script is shaped the way it is: a per-row NoC read
offset is honoured at 64 B granularity from a DRAM source and 16 B from an L1 source, and giving the
destination the source's misalignment does not help (the constraint is absolute on both addresses, not
relative). So with the L1-resident source the design already wants for pipelining, the reachable
per-row offsets are multiples of 8 bf16 elements.

This verifies the pass for the shear family whose per-row integer offset lands on those multiples:
    out[r, u] = src(8*m(r) + A*u + h(r)),   h(r) in [0, 1)
with m(r) freely chosen per row (32 distinct values here) and h(r) varying per row. That exercises
every piece of the machinery -- per-row aligned reads, row-major-to-tile conversion, the three
r-independent selection matmuls, the broadcast construction of the weight tile, floor/frac, and the
three lerps -- and it is a real 1D affine pass, not a mock.

The general case, where the integer offset is not a multiple of 8, needs the residual 0..7 element
shift handled on-chip and is the open engineering item. It is NOT verified here and is not claimed.
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
BARRIER_EVERY = 4
A = 1.31


def sel_matrices(A, nsrc):
    """P0, P1, P2: r-independent selections of source samples at floor(A*u), +1 and +2.

    out[r, u] = sum_j srcp[r, j] * P[j, u], so the matmul contraction does the selection natively.
    """
    P = [np.zeros((nsrc, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(A * u))
        for d in range(3):
            if 0 <= k + d < nsrc:
                P[d][k + d, u] = 1.0
    return P


def build(dev, ins, mode, nblocks, offs_bytes, row0):
    x, sel, frac, out = ins
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

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
    rrt[0][0] = ([x.buffer_address(), nblocks, row0] + [int(o) for o in offs_bytes]
                 + [sel.buffer_address(), frac.buffer_address()])
    crt[0][0] = [nblocks]
    wrt[0][0] = [out.buffer_address(), nblocks, 0]
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    tensors = [x, out] if mode == 0 else [x, sel, frac, out]
    return tensors, ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_fslice.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_fslice.cpp", cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(KDIR / "writer_fslice.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(CB_SRC, TILE_B, 4 * BARRIER_EVERY * SRC_TILES),
                           cb(CB_TIL, TILE_B, 2 * SRC_TILES), cb(CB_SEL, TILE_B, 3 * SRC_TILES),
                           cb(CB_FRAC, TILE_B, 2), cb(CB_OUT, TILE_B, 2 * SRC_TILES)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"A": A, "src_w": SRC_W, "win": WIN, "barrier_every": BARRIER_EVERY, "arms": {}}
    try:
        rng = np.random.default_rng(21)
        src_i = rng.integers(-120, 120, size=(128, SRC_W)).astype(np.float32)
        src_t = torch.from_numpy(src_i).to(torch.bfloat16)
        srcn = src_t.to(torch.float64).numpy()
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(src_t.reshape(1, 1, 128, SRC_W), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, memory_config=mc)
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * SRC_TILES, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        row0 = 4
        m = np.arange(NROWS, dtype=np.int64)                 # per-row base, in units of 8 elements
        offs_el = 8 * m                                       # 8-element granularity -> 16 B aligned
        offs_by = offs_el * ELEM
        h = np.array([math.modf(0.37 * r + 0.2)[0] for r in range(NROWS)])   # per-row sub-voxel

        P = sel_matrices(A, WIN)
        selt = np.concatenate([P[d].reshape(SRC_TILES, 32, 32).reshape(-1, 32) for d in range(3)], 0)
        sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        fr = np.zeros((2, 32, 32), dtype=np.float32)
        for u in range(32):
            fr[0, 0, u] = A * u - math.floor(A * u)
        for r in range(NROWS):
            fr[1, r, :] = h[r]   # replicated across columns: add_tiles_bcast_rows reads in0 per-row
        frac = ttnn.from_torch(torch.from_numpy(fr.reshape(1, 1, 64, 32)).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        ins = (x, sel, frac, out)

        # --- mode 0: reader + tilize, bit-exact -----------------------------------------------------
        t, pd = build(dev, ins, 0, 1, offs_by, row0)
        ttnn.generic_op(t, pd)
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(out).reshape(SRC_TILES, 32, 32).to(torch.float64).numpy()
        got = np.concatenate([got[i] for i in range(SRC_TILES)], axis=1)
        exp = np.stack([srcn[row0 + r, offs_el[r]:offs_el[r] + WIN] for r in range(NROWS)])
        nbad = int((got != exp).sum())
        print(f"mode 0  window 32x{WIN}, {len(set(offs_by.tolist()))} distinct offsets: "
              f"{'BIT-EXACT' if nbad == 0 else f'{nbad} mismatches'}", flush=True)
        res["arms"]["mode0"] = {"bit_exact": nbad == 0, "mismatches": nbad}
        if nbad:
            json.dump(res, open(HERE / "fslice_pass.json", "w"), indent=1)
            return

        # --- mode 1: the pass, against an fp64 single-interpolation reference ------------------------
        ref = np.zeros((32, 32))
        for r in range(NROWS):
            for u in range(32):
                p = A * u + h[r]
                j = math.floor(p)
                f = p - j
                b0 = offs_el[r] + j
                ref[r, u] = (1 - f) * srcn[row0 + r, b0] + f * srcn[row0 + r, b0 + 1]
        t, pd = build(dev, ins, 1, 1, offs_by, row0)
        ttnn.generic_op(t, pd)
        ttnn.synchronize_device(dev)
        g1 = ttnn.to_torch(out).reshape(SRC_TILES, 32, 32)[0].to(torch.float64).numpy()
        rel = float(np.linalg.norm(g1 - ref) / max(np.linalg.norm(ref), 1e-300))
        print(f"mode 1  pass vs fp64: rel L2 {rel:.4e}  max|diff| {np.abs(g1-ref).max():.3e}  "
              f"ref rms {np.sqrt((ref**2).mean()):.2f}", flush=True)
        res["arms"]["mode1"] = {"rel_l2": rel, "max_abs_diff": float(np.abs(g1 - ref).max()),
                                "ref_rms": float(np.sqrt((ref ** 2).mean()))}
        json.dump(res, open(HERE / "fslice_pass.json", "w"), indent=1)

        # --- throughput of the verified pass --------------------------------------------------------
        NB = 2000
        outb = ttnn.from_torch(torch.zeros(1, 1, 32 * NB, 32).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        t2, pd2 = build(dev, (x, sel, frac, outb), 1, NB, offs_by, row0)
        ttnn.generic_op(t2, pd2)
        ttnn.synchronize_device(dev)
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            ttnn.generic_op(t2, pd2)
            ttnn.synchronize_device(dev)
            best = min(best, time.perf_counter() - t0)
        ns = best * 1e9 / NB
        print(f"one core: {ns:8.1f} ns per output tile per pass "
              f"({130 * 1e3 / ns:8.1f} k output-tiles/s chip if it scales)", flush=True)
        res["arms"]["throughput_1core"] = {"ns_per_output_tile": ns, "nblocks": NB}
        json.dump(res, open(HERE / "fslice_pass.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
