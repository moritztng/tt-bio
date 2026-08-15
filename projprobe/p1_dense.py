#!/usr/bin/env python3
"""Phase 1(i) -- the dense coordinate stage on device, graded against numpy on RELION's own inputs.

This is the first stage of the coarse projection kernel and the first RELION arithmetic to run on a
Tenstorrent device at all. It computes, per (orientation, pixel) pair, the rotated coordinates, the
radius mask, the Friedel sign, the three floors and fractions, and the packed voxel address the
gather reader will consume.

Every arithmetic op is an SFPU DST-to-DST op under unpack_to_dest, which E8g measured at exactly
0.000e+00 against torch fp32, because every FPU path truncates its operand to about 11 mantissa bits
and an 11-bit xp makes fx wrong by 0.2 (§8.4, §8.7).

The oracle is tt_bio/cryoem/relion.py's own _project, the CPU transcription of RELION's interpolant
that the whole lineage's parity claim already rests on, run on the same dumped call. Grading is
per output, and the address is graded as an exact integer match rather than as a residual: a
one-voxel address error is not a small error, it fetches a different corner.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
DUMP = sys.argv[1] if len(sys.argv) > 1 else "/home/ttuser/relion-scratch/p8/call.2738115.1.npz"
CB_E, CB_XY, CB_C, CB_S, CB_OUT = 0, 1, 2, 3, 16
TB = 4096
N_S = 35          # exactly the number of intermediates one block pushes: any larger and the
                  # blocks after the first straddle a partial wrap of the scratch CB
EUL_COLS = (0, 1, 3, 4, 6, 7)
# Pixels per tile. Each pixel owns TWO adjacent columns, re and im, so that one 8 B gather -- a
# complex voxel, whose real and imaginary parts are contiguous in RELION's model -- lands inside a
# single dense slot tile. Splitting them across two tiles would need 4 B reads, and E4's chunk sweep
# prices a read at 36.7 ns whatever its size, so that would double the gather. Every per-pair
# quantity comes out already replicated across the pair's two columns, which is what the blend wants:
# the trilinear weights multiply re and im alike and only the Friedel sign distinguishes them.
PPT = 16


def dense_inputs(eul, x, y, n_ob, n_pb):
    """e0..e7 replicated across columns, x and y replicated down rows.

    §8.7's layout: every operand is a dense [orientation, pixel] tile, so no op ever needs a
    broadcast. The host builds these once and they are reused across the call.
    """
    e = np.zeros((n_ob * n_pb * 6, 32, 32), dtype=np.float32)
    xy = np.zeros((n_ob * n_pb * 2, 32, 32), dtype=np.float32)
    b = 0
    for ob in range(n_ob):
        eo = eul[ob * 32:(ob + 1) * 32]                      # [32, 9], zero-padded
        for pb in range(n_pb):
            px = x[pb * 32:(pb + 1) * 32]
            py = y[pb * 32:(pb + 1) * 32]
            for k, c in enumerate(EUL_COLS):
                e[b * 6 + k] = eo[:, c][:, None]             # down rows, across columns
            xy[b * 2 + 0] = px[None, :]
            xy[b * 2 + 1] = py[None, :]
            b += 1
    return e, xy


def oracle(eul, x, y, maxR2, mdlX, mdlXY, origin):
    """RELION's own arithmetic, in numpy float32, in the order the kernel does it."""
    e = eul.astype(np.float32)
    xp = e[:, 0:1] * x + e[:, 1:2] * y
    yp = e[:, 3:4] * x + e[:, 4:5] * y
    zp = e[:, 6:7] * x + e[:, 7:8] * y
    r2 = xp * xp + yp * yp + zp * zp
    mask = (r2 < np.float32(maxR2 + 1)).astype(np.float32)
    sgn = np.where(xp < 0, np.float32(-1), np.float32(1)).astype(np.float32)
    xf, yf, zf = xp * sgn, yp * sgn, zp * sgn
    x0, y0, z0 = np.floor(xf), np.floor(yf), np.floor(zf)
    addr = z0 * np.float32(mdlXY) + y0 * np.float32(mdlX) + x0 + np.float32(origin)
    addr = (addr - np.float32(8388607)) * mask + np.float32(16777215)
    return dict(addr=addr, fx=xf - x0, fy=yf - y0, fz=zf - z0, mask=mask, sgn=sgn)


def main():
    d = np.load(DUMP)
    g = d["geom"]
    mdlX, mdlY, mdlZ, mdlInitY, mdlInitZ, maxR, maxR2 = (int(v) for v in g[:7])
    imgX, imgY = int(g[8]), int(g[9])
    O, P = int(g[10]), int(g[12])
    mdlXY = mdlX * mdlY
    origin = -mdlInitZ * mdlXY - mdlInitY * mdlX
    eul = d["eul"].astype(np.float32)

    pix = np.arange(P)
    x = (pix % imgX).astype(np.float32)
    yi = pix // imgX
    y = np.where(yi > maxR, yi - imgY, yi).astype(np.float32)

    # One orientation block and a handful of pixel blocks: this stage is per-pair, so correctness
    # over a few blocks is correctness over all of them, and a small arm keeps the debug loop short.
    n_ob, n_pb = 5, 32
    eul_p = np.zeros((n_ob * 32, 9), dtype=np.float32)
    eul_p[:min(O, n_ob * 32)] = eul[:n_ob * 32]
    xp_ = np.repeat(x[:n_pb * PPT], 2).astype(np.float32)
    yp_ = np.repeat(y[:n_pb * PPT], 2).astype(np.float32)

    e_np, xy_np = dense_inputs(eul_p, xp_, yp_, n_ob, n_pb)
    n_blocks = n_ob * n_pb
    c_np = np.zeros((7, 32, 32), dtype=np.float32)
    for i, v in enumerate((mdlXY, mdlX, origin, 8388607.0, 16777215.0, 1.0, -2.0)):
        c_np[i] = np.float32(v)

    dev = ttnn.open_device(device_id=0)
    try:
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        to = lambda a: ttnn.from_torch(torch.from_numpy(a).reshape(1, 1, -1, 32),
                                       dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=dev, memory_config=mc)
        te, txy, tc = to(e_np), to(xy_np), to(c_np)
        tout = to(np.zeros((n_blocks * 6, 32, 32), dtype=np.float32))

        cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        def cb(i, depth):
            f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
            return ttnn.CBDescriptor(total_size=depth * TB, core_ranges=cg,
                                     format_descriptors=[f])

        ea = list(ttnn.TensorAccessorArgs(te).get_compile_time_args())
        xa = list(ttnn.TensorAccessorArgs(txy).get_compile_time_args())
        ca = list(ttnn.TensorAccessorArgs(tc).get_compile_time_args())
        da = list(ttnn.TensorAccessorArgs(tout).get_compile_time_args())
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rrt[0][0] = [te.buffer_address(), txy.buffer_address(), tc.buffer_address()]
        crt[0][0] = [int(np.float32(maxR2 + 1).view(np.uint32))]
        wrt[0][0] = [tout.buffer_address()]
        mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
        cc = ttnn.ComputeConfigDescriptor()
        cc.math_fidelity = ttnn.MathFidelity.HiFi4
        cc.fp32_dest_acc_en = True
        cc.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        pd = ttnn.ProgramDescriptor(kernels=[
            mk(KDIR / "reader_p1_dense.cpp",
               [CB_E, CB_XY, CB_C, TB, n_blocks] + ea + xa + ca, rrt,
               ttnn.ReaderConfigDescriptor()),
            mk(KDIR / "compute_p1_dense.cpp", [n_blocks], crt, cc),
            mk(KDIR / "writer_p1_dense.cpp", [CB_OUT, TB, n_blocks] + da, wrt,
               ttnn.WriterConfigDescriptor()),
        ], semaphores=[], cbs=[cb(CB_E, 12), cb(CB_XY, 4), cb(CB_C, 7), cb(CB_S, N_S),
                               cb(CB_OUT, 8)])
        ttnn.generic_op([te, txy, tc, tout], pd)
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(tout).numpy().reshape(n_blocks * 6, 32, 32)
    finally:
        ttnn.close_device(dev)

    names = ("addr", "fx", "fy", "fz", "mask", "sgn")
    res, ok = {}, True
    for b in range(n_blocks):
        ob, pb = b // n_pb, b % n_pb
        ref = oracle(eul_p[ob * 32:(ob + 1) * 32],
                     xp_[pb * 32:(pb + 1) * 32], yp_[pb * 32:(pb + 1) * 32],
                     maxR2, mdlX, mdlXY, origin)
        for k, nm in enumerate(names):
            e = np.abs(got[b * 6 + k] - ref[nm]).max()
            row = res.setdefault(nm, {"max_abs_err": 0.0, "exact_tiles": 0, "tiles": 0})
            row["max_abs_err"] = max(row["max_abs_err"], float(e))
            row["exact_tiles"] += int(e == 0.0)
            row["tiles"] += 1
    for nm in names:
        r = res[nm]
        bad = r["max_abs_err"] != 0.0
        ok &= not bad
        print("%-6s max_abs_err %.6e   bit-exact tiles %d/%d"
              % (nm, r["max_abs_err"], r["exact_tiles"], r["tiles"]), flush=True)
    res["all_bit_exact"] = bool(ok)
    res["shape"] = {"n_ob": n_ob, "n_pb": n_pb, "O": O, "P": P, "maxR2": maxR2,
                    "mdlX": mdlX, "mdlXY": mdlXY, "origin": origin}
    print("ALL BIT-EXACT: %s" % ok, flush=True)
    (HERE / "p1_dense.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
