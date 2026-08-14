#!/usr/bin/env python3
"""Stage 2 of Fourier-slice projection on Blackhole: build, verify, measure.

Two arms, in the order that lets a failure be attributed:
  mode 0  the reader's per-row offsets plus the row-major-to-tile conversion, emitted unchanged.
          Checked BIT-EXACTLY against the host, because it is pure data movement: if the assembled
          window is not exactly src[row0 + r, off[r] + c] then no amount of correct arithmetic
          downstream will produce a correct slice, and this is the part that cannot be reasoned about.
  mode 1  the full 1D affine resampling pass, checked against an fp64 host reference of the SAME
          single-interpolation formula the kernel implements.

Everything the reader does is what the screens measured: 32 contiguous per-row reads of a 64-wide
window (S1e: 128 B costs 1.118x 64 B), barrier amortised over 4 assemblies (S1c: 1.44x), row-major
source so a row is one transaction (S1e: a TILE_LAYOUT source would cost 2.65x).
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
NROWS = 32
SRC_TILES = 2                 # 64-wide window
WIN = 32 * SRC_TILES          # elements per row read
ELEM = 2                      # bf16
TILE_B = 32 * 32 * ELEM
SRC_W = 512                   # source row pitch in elements
BARRIER_EVERY = 4


def sel_matrices(A: float, nsrc: int):
    """P0, P1, P2: fixed selection matrices for source samples at floor(A*u), +1, +2.

    P is (nsrc x 32): P[j, u] = 1 when j is the wanted source index for output column u. The matmul
    contracts over the source index, so out[r, u] = sum_j srcp[r, j] * P[j, u] picks exactly that
    sample. They are r-independent because the reader has already applied floor(B*r + C).
    """
    P = [np.zeros((nsrc, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(A * u))
        for d in range(3):
            j = k + d
            if 0 <= j < nsrc:
                P[d][j, u] = 1.0
    return P


def host_reference(src: np.ndarray, A: float, B: float, C: float, row0: int, t: int):
    """fp64 reference for one output tile: a SINGLE linear interpolation at A*u + B*r + C.

    Returns the expected output tile and the per-row integer offsets the reader must use. The offset
    is floor(s(r) + A*32*t) so the residual fraction stays in [0, 1) and the selection matrices can be
    shared across output tiles -- the same normalisation the kernel relies on.
    """
    out = np.zeros((32, 32), dtype=np.float64)
    offs = np.zeros(32, dtype=np.int64)
    gs = np.zeros(32, dtype=np.float64)
    for r in range(32):
        s = B * r + C + A * 32 * t
        k0 = math.floor(s)
        offs[r] = k0
        gs[r] = s - k0
        for u in range(32):
            p = A * u + gs[r]
            j = math.floor(p)
            f = p - j
            a = src[row0 + r, k0 + j] if 0 <= k0 + j < src.shape[1] else 0.0
            b = src[row0 + r, k0 + j + 1] if 0 <= k0 + j + 1 < src.shape[1] else 0.0
            out[r, u] = (1 - f) * a + f * b
    return out, offs, gs


def build(dev, tensors, mode, nblocks, offs, row0):
    x, out = tensors["src"], tensors["out"]
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    rct = ([CB_SRC, WIN * ELEM, NROWS, SRC_W * ELEM, SRC_TILES, BARRIER_EVERY]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, SRC_TILES, mode]
    wct = ([CB_OUT, TILE_B, SRC_TILES if mode == 0 else 1]
           + list(ttnn.TensorAccessorArgs(out).get_compile_time_args()))
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rrt[0][0] = [x.buffer_address(), nblocks, row0] + [int(o) * ELEM for o in offs]
    crt[0][0] = [nblocks]
    wrt[0][0] = [out.buffer_address(), nblocks, 0]
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    cbs = [cb(CB_SRC, TILE_B, 2 * BARRIER_EVERY * SRC_TILES), cb(CB_TIL, TILE_B, 2 * SRC_TILES),
           cb(CB_SEL, TILE_B, 3 * SRC_TILES), cb(CB_FRAC, TILE_B, 2),
           cb(CB_OUT, TILE_B, 2 * SRC_TILES)]
    ins = [x, out] if mode == 0 else [x, tensors["sel"], tensors["frac"], out]
    return ins, ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_fslice.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_fslice.cpp", cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(KDIR / "writer_fslice.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs)


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"src_w": SRC_W, "win": WIN, "src_tiles": SRC_TILES, "barrier_every": BARRIER_EVERY,
           "arms": {}}
    try:
        torch.manual_seed(11)
        rng = np.random.default_rng(11)
        # A row-major source. bf16 values that are exactly representable, so mode 0 can be checked for
        # bit-exactness rather than for closeness -- a movement bug must not hide behind a tolerance.
        src_i = rng.integers(-120, 120, size=(256, SRC_W)).astype(np.float32)
        src_t = torch.from_numpy(src_i).to(torch.bfloat16)
        srcn = src_t.to(torch.float64).numpy()
        x = ttnn.from_torch(src_t.reshape(1, 1, 256, SRC_W), dtype=ttnn.bfloat16,
                            layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
        A, B, C, row0, t = 1.31, 0.47, 3.2, 8, 0
        ref, offs, gs = host_reference(srcn, A, B, C, row0, t)

        outn = ttnn.from_torch(torch.zeros(1, 1, 32 * SRC_TILES, 32).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        # Selection matrices, and the two fraction vectors: frac(A*u) in row 0, g(r) in column 0.
        P = sel_matrices(A, WIN)
        selt = np.concatenate([P[d].reshape(SRC_TILES, 32, 32).transpose(0, 1, 2).reshape(-1, 32)
                               for d in range(3)], axis=0)
        sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        fr = np.zeros((2, 32, 32), dtype=np.float32)
        for u in range(32):
            fr[0, 0, u] = A * u - math.floor(A * u)
        for r in range(32):
            fr[1, r, 0] = gs[r]
        frac = ttnn.from_torch(torch.from_numpy(fr.reshape(1, 1, 64, 32)).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        tensors = {"src": x, "out": outn, "sel": sel, "frac": frac}

        # --- mode 0: the reader and the tilize, bit-exact ------------------------------------------
        ins, pd = build(dev, tensors, 0, 1, offs, row0)
        ttnn.generic_op(ins, pd)
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(outn).reshape(SRC_TILES, 32, 32).to(torch.float64).numpy()
        got = np.concatenate([got[i] for i in range(SRC_TILES)], axis=1)   # 32 x 64 logical window
        exp = np.stack([srcn[row0 + r, offs[r]:offs[r] + WIN] for r in range(32)])
        nbad = int((got != exp).sum())
        print(f"mode 0  assembled window 32x{WIN}: {'BIT-EXACT' if nbad == 0 else f'{nbad} mismatches'}"
              f"   max|diff| {np.abs(got - exp).max():.3e}", flush=True)
        res["arms"]["mode0_tilize"] = {"bit_exact": nbad == 0, "mismatches": nbad,
                                       "max_abs_diff": float(np.abs(got - exp).max())}
        json.dump(res, open(HERE / "fslice_build.json", "w"), indent=1)
        if nbad:
            print("  reader/tilize is wrong; not proceeding to the arithmetic arm.", flush=True)
            np.set_printoptions(linewidth=200, suppress=True)
            print("  offs[:6]      ", offs[:6])
            print("  exp[0,:8]     ", exp[0, :8])
            print("  got[0,:8]     ", got[0, :8])
            print("  exp[1,:8]     ", exp[1, :8])
            print("  got[1,:8]     ", got[1, :8])
            print("  src[row0,:12] ", srcn[row0, :12])
            print("  src[row0+1,:12]", srcn[row0 + 1, :12])
            # where does got[0,0] appear in the source?
            hit = np.argwhere(srcn == got[0, 0])
            print("  got[0,0] =", got[0, 0], "found at (row,col):", hit[:6].tolist())
            hit2 = np.argwhere(srcn == got[0, 1])
            print("  got[0,1] =", got[0, 1], "found at (row,col):", hit2[:6].tolist())
            return

        # --- mode 1: the full pass, against the fp64 single-interpolation reference ----------------
        ins, pd = build(dev, tensors, 1, 1, offs, row0)
        ttnn.generic_op(ins, pd)
        ttnn.synchronize_device(dev)
        got1 = ttnn.to_torch(outn).reshape(SRC_TILES, 32, 32)[0].to(torch.float64).numpy()
        den = max(np.linalg.norm(ref), 1e-300)
        rel = float(np.linalg.norm(got1 - ref) / den)
        print(f"mode 1  full pass vs fp64 reference: rel L2 {rel:.4e}  "
              f"max|diff| {np.abs(got1 - ref).max():.4e}  ref rms {np.sqrt((ref**2).mean()):.2f}",
              flush=True)
        res["arms"]["mode1_pass"] = {"rel_l2": rel,
                                     "max_abs_diff": float(np.abs(got1 - ref).max()),
                                     "ref_rms": float(np.sqrt((ref ** 2).mean()))}
        json.dump(res, open(HERE / "fslice_build.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
