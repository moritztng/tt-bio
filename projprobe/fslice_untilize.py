#!/usr/bin/env python3
"""Does pack_untilize_dest work at all in this kernel? Isolation test, no arithmetic.

Mode 7 (row-major output, needed to chain stage 2a into 2b) returns rel L2 ~1.0-1.1 against fp64 while
the tiled path returns 3.39e-3 on identical inputs. Two suspects were named: the packer dest-offset
state shared with the tilize reconfiguration, and the face_r_dim/num_faces defaults. Debugging mode 7
directly cannot separate those from a bug in how mode 7 arranges its DST.

Mode 8 removes everything else: tilize the assembled window, copy tile 0 straight into DST 0, untilize
it back out. The answer must be the first 32 columns of the row-major window the reader assembled,
which is known exactly. Pass means the untilize is fine and mode 7 misuses it; fail means the untilize
is broken in this context and mode 7 is not worth debugging further.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent / "tt_bio" / "kernels" / "fslice"
CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, CB_MID = 0, 4, 8, 12, 16, 20
NROWS, SRC_TILES = 32, 2
WIN = 32 * SRC_TILES
ELEM = 2
TILE_B = 32 * 32 * ELEM
SRC_W, SRC_ROWS = 1024, 64
BARRIER_EVERY = 4


def build(dev, x, sel, frac, out, offs_bytes, rowidx, mode):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    rct = ([CB_SRC, WIN * ELEM, NROWS, SRC_W * ELEM, SRC_TILES, BARRIER_EVERY, mode,
            CB_SEL, CB_FRAC, 3 * SRC_TILES, TILE_B, 3]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(frac).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, SRC_TILES, mode, CB_MID]
    wct = [CB_OUT, TILE_B, 1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rrt[0][0] = ([x.buffer_address(), 1, 0] + [int(o) for o in offs_bytes]
                 + [sel.buffer_address(), frac.buffer_address()] + [int(i) for i in rowidx])
    crt[0][0] = [1]
    wrt[0][0] = [out.buffer_address(), 1, 0]
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
                           cb(CB_FRAC, TILE_B, 4), cb(CB_OUT, TILE_B, 4), cb(CB_MID, TILE_B, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        rng = np.random.default_rng(41)
        base = rng.integers(-120, 120, size=(SRC_ROWS, SRC_W)).astype(np.float32)
        basen = torch.from_numpy(base).to(torch.bfloat16).to(torch.float64).numpy()
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(torch.from_numpy(base).to(torch.bfloat16).reshape(1, 1, SRC_ROWS, SRC_W),
                            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                            memory_config=mc)
        dummy = ttnn.from_torch(torch.zeros(1, 1, 96, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)
        # Identity in tile 0, zeros elsewhere: srcp0 . I + srcp1 . 0 = srcp0, so mode 10's expected
        # answer is the same window mode 8 produces and the matmul is the only thing that changed.
        seln = np.zeros((96, 32), dtype=np.float32)
        seln[:32, :] = np.eye(32, dtype=np.float32)
        sel_id = ttnn.from_torch(torch.from_numpy(seln).to(torch.bfloat16).reshape(1, 1, 96, 32),
                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        offs_el = 8 * np.arange(NROWS, dtype=np.int64)
        offs_by = offs_el * ELEM
        rowidx = np.arange(NROWS, dtype=np.int64)
        exp = np.stack([basen[r, offs_el[r]:offs_el[r] + 32] for r in range(NROWS)])

        for mode, want, label in ((8, exp, "copy_tile -> untilize          "),
                                  (9, 2.0 * exp, "copy_tile + 1 SFPU -> untilize"),
                                  (10, exp, "MATMUL(identity) -> untilize  ")):
            out = ttnn.from_torch(torch.zeros(1, 1, 1, 1024).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                  layout=ttnn.ROW_MAJOR_LAYOUT, device=dev)
            pd = build(dev, x, sel_id, dummy, out, offs_by, rowidx, mode)
            ins = [x, sel_id, dummy, out]
            ttnn.generic_op(ins, pd)
            ttnn.synchronize_device(dev)
            g = ttnn.to_torch(out).reshape(32, 32).to(torch.float64).numpy()
            nb = int((g != want).sum())
            print(f"mode {mode} {label}: "
                  f"{'BIT-EXACT' if nb == 0 else str(nb) + ' mismatches'}"
                  f"   max|diff| {np.abs(g - want).max():.3e}", flush=True)
            res[f"mode{mode}_bit_exact"] = nb == 0
            res[f"mode{mode}_mismatches"] = nb
            ttnn.deallocate(out)
        json.dump(res, open(HERE / "fslice_untilize.json", "w"), indent=1)
        g = np.zeros((32, 32))
        nbad = 0
        if False:
            pass
        if nbad:
            np.set_printoptions(linewidth=200, suppress=True)
            print("  exp[0,:8]", exp[0, :8])
            print("  got[0,:8]", g[0, :8])
            print("  exp[1,:8]", exp[1, :8])
            print("  got[1,:8]", g[1, :8])
            hit = np.argwhere(exp == g[0, 0])
            print(f"  got00={g[0,0]} appears in exp at {hit[:5].tolist()}")
        res["untilize_bit_exact"] = nbad == 0
        res["untilize_mismatches"] = nbad
        json.dump(res, open(HERE / "fslice_untilize.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
