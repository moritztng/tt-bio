#!/usr/bin/env python3
"""The integrated adjoint arm: stage 0', stage 2' and stage 1' chained on device, one timed region.

Backprojection run DESTINATION-STATIONARY, which is what makes the no-scatter claim pay:

  S0'  DRAM -> L1     one bulk read of the slice, then the 8 sub-offset copies S2's reader indexes.
                      This IS the forward's stage 1 with the band collapsed to a single plane:
                      reader_zcollapse + compute_zcollapse_rm + writer_strip, nplane = 1, mask = 1.0,
                      shift_real = 1. Not one line of new kernel code, which is the "80% of the
                      machinery" claim being paid in the most literal way available.
  S2'  slice -> W     W'[r, j] = sum_d ( c_d(r, u) * y[r, u] ) . P_d^T[u, j]
                      Destination = one ALIGNED W tile. Its contributions are 32 rows of the slice,
                      each at its own offset, which is the forward's own reader unchanged
                      (reader_fslice.cpp) pointed at the L1-resident replicated slice instead of an
                      L1-resident plane. The write is one bulk page.
  S1'  W -> volume    V'[X, Y, z] = sum_{d: band covers z} mask[z - z0_d][X, Y] * W_d[X, Y]
                      Destination = one volume tile. Every contributing direction multiplies into the
                      same DST slot under one acquire; the tile is written once, in bulk.

The SOURCE-stationary form is the one that needs a scatter, and fslice_bproj.py measured that form:
fix the slice tile, produce the 64-wide W window, write it at 32 per-row offsets. Its 0.98x-of-bulk
at eight batched contributions is measuring the wrong thing, because the batched contributions are
different psi and psi is what sets the offsets -- they do not share a destination window and cannot
be packed once.

THE WINDOW WIDTH IS WHERE THE ADJOINT IS NOT THE FORWARD. The forward reads a 64-wide W window to
make one 32-wide slice tile, because the resample stretches by A = 1.31 and the read window may sit
anywhere. The adjoint's destination must be tile-ALIGNED, and an aligned 32-wide destination is fed
by a u-run of 32/A = 24.4, so its source window is 32 wide, not 64. Same 32 transactions per
assembly, half the bytes each.

WHERE THE DRAM TRAFFIC IS. Every buffer this arm touches is DRAM-resident except the replicated slice
store, which is L1 by construction (a per-row read offset is quantised to 64 B from DRAM and 16 B
from L1, so the offsets are only expressible at all out of L1). So:
  S0'  reads 2*box^2 B of slice from DRAM     <- the floor's read term, paid here and nowhere else
       writes 8 * 2*box^2 B into L1           <- the misalignment tax, section 8.5
  S2'  reads the assemblies from L1, writes one 2048 B W page per destination tile to DRAM
  S1'  reads W pages from DRAM, writes one volume page per destination tile to DRAM

ACCUMULATION PRECISION. fp32_dest_acc_en is ON for both compute stages and every CB either kernel
packs into is bf16. That combination is not a free choice: bf16 L1 with an fp32 accumulator is what
works, fp32 L1 with a bf16 accumulator is a factor of 9 worse at depth 48, and fp32 L1 with an fp32
accumulator silently drops one contribution per acquire (bproj_s2_diag.py, and the two faults are
root-caused in state/relion-backprojection.md section 8.3).

AMORTISATION, every constant stated rather than assumed.
  n_orient = 768 HEALPix order-3 directions x 96 psi = 73,728 slices per iteration.
  S0'      one slice staged per box/2 rows, i.e. (box/2)/32 strips, each strip_tiles = row_el/32
           tiles wide. One DRAM read per slice, 8 L1 copies out of it.
  S2'      one aligned W tile per block, `ncontrib` contributions accumulated into it. The (W tile,
           slice) incidence count per slice is the slice's own tile count, pi*(box/2)^2/2/1024 per
           component x 2 components, because the shear preserves area. `ncontrib` is therefore a
           PARTICLE COUNT in disguise: contributions per W tile = Np * inc_per_slice / (768 * 2 *
           (2box/32)^2), so ncontrib = 48 at box 256 is a 376k-particle dataset and ncontrib = 16 is
           125k. It sets only how far the W-store write amortises, and the W store is not in the
           floor -- it is this decomposition's own intermediate, priced as a named residual.
  S1'      one volume tile per block. The padded half-space accumulator is (2box)^2 x (box+1)
           voxels x 3 components (complex data + REAL WEIGHT), written once per iteration, so
           3*(2box)^2*(box+1)/1024/73,728 tile writes per slice. Running nwin = 28 contributions on
           every one of those blocks OVER-prices S1' by 1.7x against the true (direction, volume
           tile) incidence count, because the padded corners carry a write but no contribution. The
           measured rate is therefore a lower bound, deliberately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE.parent / "tt_bio" / "kernels" / "fslice"

# S0' CBs / S2' CBs / S1' CBs (disjoint programs, never resident at once)
CB_V, CB_SMASK, CB_MID1, CB_OUT1, CB_MID2 = 0, 4, 8, 12, 16
CB_SRC, CB_TIL, CB_COEF, CB_SELT, CB_MID, CB_ACC, CB_OUT = 0, 4, 8, 12, 16, 20, 24
CB_W, CB_MASK, CB_VOUT = 0, 4, 8

ELEM = 2                      # the slice crosses the NoC as bf16
TILE_B = 32 * 32 * ELEM
F32_TILE_B = 32 * 32 * 4
NROWS, SRC_TILES = 32, 1
WIN = 32 * SRC_TILES
NCOPY = 8
NPLANE = 28                   # S3's mean band over real HEALPix directions
PSI = 96
NDIR = 768                    # HEALPix order 3
NORIENT = NDIR * PSI          # 73,728
A = 1.31
BARRIER_EVERY = 4
NX, NY = 13, 10
WPAGES = 2048                 # W store pages the S1' reader cycles

# Measured on THIS card by projprobe/b0_roofs.py: 130 cores, 2048 B pages, disjoint page ranges,
# write arm verified by reading the pattern back. Nothing inherited from the projection pass.
READ_ROOF = 404.5e9
WRITE_ROOF = 173.5e9

tiles_per_slice = lambda box: math.pi * (box / 2.0) ** 2 / 2.0 / 1024.0


def vol_bytes(box):
    """Padded half-space accumulator: complex data + real weight, fp32. The weight is a third."""
    return 3 * (2 * box) ** 2 * (box + 1) * 4


def vol_tiles_per_slice(box):
    return 3 * (2 * box) ** 2 * (box + 1) / 1024.0 / NORIENT


def floor_slices_s(box):
    """T = slice read / READ_ROOF + volume write per slice / WRITE_ROOF. Two roofs, measured apart."""
    return 1.0 / (2 * box * box / READ_ROOF + vol_bytes(box) / NORIENT / WRITE_ROOF)


def slice_geom(box):
    """A half-space slice as this pipeline lays it out, and the strips S0' stages it in.

    row_el = 2*box elements is the padded plane row (box complex pairs), and a slice is box/2 of
    them, so a slice is exactly 2*box^2 BYTES at bf16 -- the floor's read term.
    """
    row_el = 2 * box
    strip_tiles = row_el // 32
    strips_per_slice = (box // 2) // 32
    return row_el, strip_tiles, strips_per_slice


def shift_matrices():
    """S_q and S'_q for q = 0..NCOPY-1: copy q of a strip is the strip shifted left by q."""
    out = []
    for q in range(NCOPY):
        s0 = np.zeros((32, 32), dtype=np.float32)
        s1 = np.zeros((32, 32), dtype=np.float32)
        for c in range(32):
            if c + q < 32:
                s0[c + q, c] = 1.0
            else:
                s1[c + q - 32, c] = 1.0
        out += [s0, s1]
    return np.stack(out)


def selt_matrices(a):
    """P_d^T restricted to one aligned destination tile: (u, j), u and j both 0..31."""
    q = [np.zeros((32, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(a * u))
        for d in range(3):
            if 0 <= k + d < 32:
                q[d][u, k + d] = 1.0
    return q


def coef_tiles(h):
    c = np.zeros((3, 32, 32), dtype=np.float32)
    for r in range(NROWS):
        for u in range(32):
            w = h[r] + (A * u - math.floor(A * u))
            m = math.floor(w)
            wq = w - m
            c[0, r, u] = (1 - wq) * (1 - m)
            c[1, r, u] = (1 - wq) * m + wq * (1 - m)
            c[2, r, u] = wq * m
    return c


def masks():
    a, b = 0.43, 0.29
    mk_ = np.zeros((NPLANE, 32, 32), dtype=np.float32)
    for x in range(32):
        for y in range(32):
            z = a * x + b * y
            z0 = int(np.floor(z))
            t = z - z0
            if 0 <= z0 < NPLANE:
                mk_[z0, x, y] = 1 - t
            if 0 <= z0 + 1 < NPLANE:
                mk_[z0 + 1, x, y] = t
    return mk_


def shear(bs, cs):
    s_r = bs * np.arange(NROWS) + cs
    k0 = np.floor(s_r).astype(np.int64)
    h = s_r - k0
    return k0, h, k0 % NCOPY, 8 * (k0 // NCOPY)


def mk(p, cg, ct, rt, cfg):
    return ttnn.KernelDescriptor(kernel_source=str(p),
                                 source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                 core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)


def cbs(cg, spec):
    out = []
    for i, fmt, page, depth in spec:
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=page)
        out.append(ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg,
                                     format_descriptors=[f]))
    return out


def grid(nx, ny):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])


def build_s0(dsl, s0m, sl, nx, ny, nstrip, row_el, strip_tiles, npages_per_core, nmask,
             shift_real=1, hoist=2, fid=ttnn.MathFidelity.HiFi2):
    """S0': DRAM slice -> the 8-copy L1 replicated store.

    This is `build_stage1` from projprobe/fslice_e2e.py with nplane = 1 and the mask set to 1.0, so
    the z-collapse degenerates to a pass-through and what is left is exactly the replication: tilize
    the strip, produce copy q as two matmuls against the fixed 0/1 shift matrices, untilize each copy
    back to row-major and write it at the destination's row pitch. Every kernel is the forward's,
    unmodified.

    Each core owns a DISJOINT contiguous DRAM range of `npages_per_core` pages and reads it once, so
    the arm's read pattern is the one b0_roofs.py measured the 404.5 GB/s read roof with rather than
    130 cores re-reading each other's pages.
    """
    cg = grid(nx, ny)
    ncore = nx * ny
    rct = ([CB_V, CB_SMASK, 1, TILE_B, BARRIER_EVERY, 1, nmask]
           + list(ttnn.TensorAccessorArgs(dsl).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(s0m).get_compile_time_args()))
    # hoist 2 is the two-sweep form: shift the whole copy, then untilize the whole copy, so the
    # packer is configured once per copy instead of once per tile. hoist 1 is WRONG with shift_real on
    # (rel L2 1.2 on the identity copy) and hoist 0 pays a pack_untilize_dest_init MMIO per tile.
    # HiFi2 rather than HiFi4: the shift operand is an exact 0/1 permutation and the data is bf16, so
    # HiFi2's 10 mantissa bits cover bf16's 8 exactly -- verified by the S0' parity staying 0.0.
    cct = [CB_V, CB_SMASK, CB_MID1, CB_OUT1, 1, 1, NCOPY, strip_tiles, hoist, shift_real, CB_MID2]
    wct = ([CB_OUT1, row_el * ELEM, 32, NCOPY, strip_tiles, ncore]
           + list(ttnn.TensorAccessorArgs(sl).get_compile_time_args()))
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = [dsl.buffer_address(), s0m.buffer_address(), nstrip * strip_tiles,
                           c * npages_per_core, ncore * npages_per_core]
            crt[cx][cy] = [nstrip]
            wrt[cx][cy] = [sl.buffer_address(), nstrip, c]
            c += 1
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_zcollapse.cpp", cg, rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_zcollapse_rm.cpp", cg, cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=fid)),
        mk(KDIR / "writer_strip.cpp", cg, wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs(cg, [
        (CB_V, ttnn.bfloat16, TILE_B, 2),
        (CB_SMASK, ttnn.bfloat16, TILE_B, nmask),
        (CB_MID1, ttnn.bfloat16, TILE_B, 2 * strip_tiles),
        (CB_OUT1, ttnn.bfloat16, TILE_B, 2 * strip_tiles),
        # hoist 2 shifts a whole strip before it untilizes any of it, so cb_mid2 is strip-deep.
        (CB_MID2, ttnn.bfloat16, TILE_B, (2 * strip_tiles if hoist == 2 else 2))]))


def build_s2a(sl, coef, selt, w, nx, ny, nb, row_el, offs_bytes, rowidx_of_core, ncontrib,
              chunk, compute="compute_bproj_ds.cpp", mid=None, dstacc=True):
    """S2': the forward's stage-2 reader and writer, UNCHANGED, around the adjoint's compute.

    fp32_dest_acc_en is decoupled from the CB format on purpose: deriving one from the other is
    exactly the combination that loses a contribution per acquire (section 8.3). Every CB the compute
    kernel packs into is bf16 and the accumulator is fp32 DST.
    """
    cg = grid(nx, ny)
    nselt = 3 * SRC_TILES
    rct = ([CB_SRC, WIN * ELEM, NROWS, row_el * ELEM, SRC_TILES, BARRIER_EVERY, 13,
            CB_SELT, CB_COEF, nselt, TILE_B, 3]
           + list(ttnn.TensorAccessorArgs(sl).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(selt).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(coef).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_COEF, CB_SELT, CB_MID, CB_ACC, CB_OUT, SRC_TILES, ncontrib, chunk]
    wct = [CB_OUT, TILE_B, 1] + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = ([sl.buffer_address(), nb * ncontrib, 0] + [int(o) for o in offs_bytes]
                           + [selt.buffer_address(), coef.buffer_address()]
                           + [int(i) for i in rowidx_of_core[c]])
            crt[cx][cy] = [nb]
            wrt[cx][cy] = [w.buffer_address(), nb, c * nb]
            c += 1
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_fslice.cpp", cg, rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / compute, cg, cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                        fp32_dest_acc_en=dstacc)),
        mk(KDIR / "writer_fslice.cpp", cg, wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs(cg, [
        (CB_SRC, ttnn.bfloat16, TILE_B, 4 * BARRIER_EVERY * SRC_TILES),
        # Every CB this kernel PACKS into carries ONE format, bf16, and DST accumulates in fp32.
        # cb_mid at bf16 against cb_acc/cb_out at fp32 dropped the first tile packed after each
        # mm_init -- one contribution per acquire -- and pack_reconfig_data_format did not cover it.
        (CB_TIL, ttnn.bfloat16, TILE_B, 2 * SRC_TILES),
        (CB_COEF, ttnn.bfloat16, TILE_B, 3),
        (CB_SELT, ttnn.bfloat16, TILE_B, nselt),
        (CB_MID, mid or ttnn.bfloat16, TILE_B, 2 * 3 * SRC_TILES * chunk),
        (CB_ACC, ttnn.bfloat16, TILE_B, 2),
        (CB_OUT, ttnn.bfloat16, TILE_B, 2)]))


def build_s1a(w, m, vol, nx, ny, nb, nstep, vol_fmt, vol_page, wpages, dstacc=True):
    """S1': destination-stationary z-spread, sliding window over the contributing directions.

    W and the masks are bf16 -- W because S2' produced it that way, the masks because they are the
    forward's own bf16 masks and the reader loads both operands at one page size. Only the volume
    accumulator is fp32, and it is the only CB this kernel packs into, so section 8.3's one-format
    rule is satisfied trivially.
    """
    cg = grid(nx, ny)
    nwin = NPLANE * nstep
    rct = ([CB_W, CB_MASK, NPLANE, nstep, TILE_B, BARRIER_EVERY]
           + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(m).get_compile_time_args()))
    cct = [CB_W, CB_MASK, CB_VOUT, NPLANE, nstep]
    wct = [CB_VOUT, vol_page, 1] + list(ttnn.TensorAccessorArgs(vol).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = [w.buffer_address(), m.buffer_address(), nb, (c * 29) % wpages, wpages]
            crt[cx][cy] = [nb]
            wrt[cx][cy] = [vol.buffer_address(), nb, c * nb]
            c += 1
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_zspread_acc.cpp", cg, rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_zspread_acc.cpp", cg, cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                        fp32_dest_acc_en=dstacc)),
        mk(KDIR / "writer_fslice.cpp", cg, wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs(cg, [
        (CB_W, ttnn.bfloat16, TILE_B, nwin + nstep * BARRIER_EVERY * 2),
        (CB_MASK, ttnn.bfloat16, TILE_B, NPLANE),
        (CB_VOUT, vol_fmt, vol_page, 2)]))


def s0_reference(dsln, strip_tiles, group):
    """The strip core 0 last staged, and its 8 shifted copies, in fp64.

    Copy q of a strip is the strip shifted LEFT by q elements; past the end of the strip there is
    only pad, which is why the last tile of a strip drops its second matmul term.
    """
    base = group * strip_tiles
    strip = np.concatenate([dsln[base + t] for t in range(strip_tiles)], axis=1)
    w = strip.shape[1]
    out = np.zeros((NCOPY, 32, w))
    for q in range(NCOPY):
        out[q, :, :w - q] = strip[:, q:]
    return strip, out


def s0_parity(dev, res, dsl, s0m, sl, row_el, strip_tiles, dsln, npages_per_core, nmask,
              shift_real=1, hoist=2, fid=None):
    """One core, one strip: does the replicated store hold the real sub-offset shifts?"""
    pd = build_s0(dsl, s0m, sl, 1, 1, 1, row_el, strip_tiles, npages_per_core, nmask, shift_real,
                  hoist, fid)
    ttnn.generic_op([dsl, s0m, sl], pd)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(sl)[0, 0, :32 * NCOPY, :].to(torch.float64).numpy()
    _, ref = s0_reference(dsln, strip_tiles, 0)
    err, worst = [], 0.0
    for q in range(NCOPY):
        g = got[q::NCOPY][:32]
        e = float(np.linalg.norm(g - ref[q]) / max(np.linalg.norm(ref[q]), 1e-300))
        err.append(e)
        worst = max(worst, e)
    print("  S0' 8 sub-offset copies vs fp64: rel L2 per copy "
          + " ".join(f"{e:.1e}" for e in err), flush=True)
    res["parity_s0_rel_l2"] = err
    res["parity_s0_worst"] = worst
    return worst


def s2_reference(sln, cfn, qn, rho, offs_el, ncontrib):
    """fp64 adjoint of exactly what the reader assembled, accumulated over ncontrib contributions."""
    win = np.stack([sln[r * NCOPY + rho[r], offs_el[r]:offs_el[r] + WIN] for r in range(NROWS)])
    one = np.zeros((32, 32))
    for d in range(3):
        one += (cfn[d] * win) @ qn[d]
    return one * ncontrib, win


def s2_parity(dev, res, chunk, row_el, sl, sln, coef, cfn, selt, qn, offs_el, rho, nc2, wpg):
    """One core, one W tile, nc2 contributions. A scale-only residual means contributions are being
    LOST rather than computed wrong, and the least-squares scale says how many of them landed."""
    w1 = ttnn.from_torch(torch.zeros(1, 1, 32 * wpg, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    rowidx1 = [(np.arange(NROWS) * NCOPY + rho).tolist()]
    pd = build_s2a(sl, coef, selt, w1, 1, 1, 1, row_el, offs_el * ELEM, rowidx1, nc2, chunk)
    ttnn.generic_op([sl, coef, selt, w1], pd)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(w1)[0, 0, :32, :].to(torch.float64).numpy()
    ref, _ = s2_reference(sln, cfn, qn, rho, offs_el, nc2)
    one = ref / nc2
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300))
    k = float((got * one).sum() / (one * one).sum())
    resid = float(np.linalg.norm(got - k * one) / max(np.linalg.norm(got), 1e-300))
    err = np.abs(got - ref).ravel()
    print("  S2' nc %3d (%d chunk): rel L2 %.4e   landed %7.3f of %d   resid %.3e   "
          "err p50 %.2e p99 %.2e max %.2e"
          % (nc2, max(1, nc2 // chunk), rel, k, nc2, resid, np.percentile(err, 50),
             np.percentile(err, 99), err.max()), flush=True)
    res["parity_s2_nc%d" % nc2] = {"rel_l2": rel, "landed": k, "resid": resid,
                                   "err_pct": {q: float(np.percentile(err, q))
                                               for q in (50, 90, 99, 100)}}
    ttnn.deallocate(w1)
    return rel


def s1_parity(dev, res, wsrc, m_t, nstep, vol_fmt, vol_page, tdt, mkn, tag, wpages):
    """One core, one volume tile, nwin contributions off the DEVICE's own W."""
    nwin = NPLANE * nstep
    v1 = ttnn.from_torch(torch.zeros(1, 1, 32 * 4, 32).to(tdt), dtype=vol_fmt,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    pd = build_s1a(wsrc, m_t, v1, 1, 1, 1, nstep, vol_fmt, vol_page, wpages)
    ttnn.generic_op([wsrc, m_t, v1], pd)
    ttnn.synchronize_device(dev)
    gv = ttnn.to_torch(v1)[0, 0, :32, :].to(torch.float64).numpy()
    wn = ttnn.to_torch(wsrc).reshape(-1, 32, 32).to(torch.float64).numpy()
    npg = wpages
    refv = np.zeros((32, 32))
    for i in range(nwin):
        refv += wn[i % npg] * mkn[(nwin - 1 - i) // nstep]
    relv = float(np.linalg.norm(gv - refv) / max(np.linalg.norm(refv), 1e-300))
    ev = np.abs(gv - refv).ravel()
    print(f"  S1' W -> volume tile vs fp64, {nwin:3d} contributions, {tag}: rel L2 {relv:.4e}  "
          f"p50 {np.percentile(ev,50):.2e}  p99 {np.percentile(ev,99):.2e}  max {ev.max():.2e}",
          flush=True)
    res["parity_s1_%s_nwin%d" % (tag, nwin)] = {
        "rel_l2": relv, "err_pct": {p: float(np.percentile(ev, p)) for p in (50, 90, 99, 100)}}
    ttnn.deallocate(v1)
    return relv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("box", type=int, nargs="?", default=256)
    ap.add_argument("--ncontrib", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--nstep", type=int, default=1)
    ap.add_argument("--nb2", type=int, default=8)
    ap.add_argument("--acc", choices=["fp32", "bf16"], default="fp32",
                    help="the VOLUME accumulator and both DST accumulators; L1 is always bf16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--no-s0", action="store_true", help="the pass-2 arm: slice store never staged")
    ap.add_argument("--hoist", type=int, default=2, choices=[0, 1, 2],
                    help="S0' packer init: 0 per tile, 1 per strip (WRONG with real shifts), "
                         "2 two-sweep, once per copy")
    ap.add_argument("--s0-fid", choices=["HiFi2", "HiFi4"], default="HiFi2")
    ap.add_argument("--shift-real", type=int, default=1,
                    help="0 = the copies carry the right bytes and the wrong contents, which prices\n                         the replication's L1 traffic apart from its shift matmuls")
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    assert a.ncontrib % a.chunk == 0
    dstacc = a.acc == "fp32"
    vol_fmt = ttnn.float32 if dstacc else ttnn.bfloat16
    vol_page = F32_TILE_B if dstacc else TILE_B
    tdt = torch.float32 if dstacc else torch.bfloat16
    fid = getattr(ttnn.MathFidelity, a.s0_fid)
    row_el, strip_tiles, strips_per_slice = slice_geom(a.box)
    ncore = NX * NY
    nwin = NPLANE * a.nstep

    dev = ttnn.open_device(device_id=0)
    res = {"box": a.box, "row_el": row_el, "ncontrib": a.ncontrib, "chunk": a.chunk,
           "nstep": a.nstep, "nwin": nwin, "acc": a.acc, "ncore": ncore, "src_tiles": SRC_TILES,
           "strip_tiles": strip_tiles, "shift_real": a.shift_real, "hoist": a.hoist,
           "s0_fid": a.s0_fid, "strips_per_slice": strips_per_slice, "s0": not a.no_s0,
           "read_roof_gbs": READ_ROOF / 1e9, "write_roof_gbs": WRITE_ROOF / 1e9,
           "floor_slices_s": floor_slices_s(a.box)}
    try:
        # --- the L1 replicated slice store: ncore strips x 32 rows x 8 copies -------------------
        sl_rows = ncore * 32 * NCOPY
        sl = ttnn.from_torch(torch.zeros(1, 1, sl_rows, row_el).to(torch.bfloat16),
                             dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                             memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED,
                                                             ttnn.BufferType.L1))
        res["slice_l1_mb"] = sl_rows * row_el * ELEM / 2 ** 20

        k0, h, rho, offs_el = shear(0.77, 5.3)
        assert len(set(rho.tolist())) == NCOPY
        cf = coef_tiles(h)
        coef = ttnn.from_torch(torch.from_numpy(cf.reshape(1, 1, 96, 32)).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        cfn = torch.from_numpy(cf).to(torch.bfloat16).to(torch.float64).numpy()

        q = selt_matrices(A)
        selt_np = np.concatenate(q, axis=0).astype(np.float32)
        selt = ttnn.from_torch(torch.from_numpy(selt_np).to(torch.bfloat16).reshape(1, 1, -1, 32),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        qn = torch.from_numpy(selt_np).to(torch.bfloat16).to(torch.float64).numpy().reshape(3, 32, 32)

        mkf = masks()
        m_t = ttnn.from_torch(torch.from_numpy(mkf.reshape(1, 1, NPLANE * 32, 32)).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        mkn = torch.from_numpy(mkf).to(torch.bfloat16).to(torch.float64).numpy()

        rowidx_of_core = [((c * 32 + np.arange(NROWS)) * NCOPY + rho).tolist() for c in range(ncore)]

        # --- how many slices the arm processes, and how much of each stage that implies ----------
        inc_per_slice = tiles_per_slice(a.box) * 2
        nslice = a.nb2 * ncore * a.ncontrib / inc_per_slice
        nb1 = max(1, int(round(nslice * vol_tiles_per_slice(a.box) / ncore)))
        # S0' must stage every one of those slices from DRAM: strips_per_slice strips each.
        nstrip = max(1, int(math.ceil(nslice * strips_per_slice / ncore)))
        npages_per_core = nstrip * strip_tiles
        npages = ncore * npages_per_core
        # contributions per W tile as a particle count, so ncontrib is not a free parameter
        w_tiles_total = NDIR * 2 * (row_el // 32) ** 2
        nparticle = a.ncontrib * w_tiles_total / inc_per_slice
        res.update({"nb2": a.nb2, "nb1": nb1, "nslice": nslice, "inc_per_slice": inc_per_slice,
                    "vol_tiles_per_slice": vol_tiles_per_slice(a.box), "nstrip": nstrip,
                    "npages_per_core": npages_per_core, "npages": npages,
                    "dram_src_mb": npages * TILE_B / 2 ** 20, "nparticle_implied": nparticle})

        # --- the DRAM slice source: each core owns a disjoint contiguous range ------------------
        rng = np.random.default_rng(97)
        nbase = min(npages, 2048)
        base = rng.integers(-100, 100, size=(nbase * 32, 32)).astype(np.float32)
        bt = torch.from_numpy(base).to(torch.bfloat16)
        reps_ = int(math.ceil(npages / nbase))
        dsl_t = bt.repeat(reps_, 1)[:npages * 32].contiguous()
        dsl = ttnn.from_torch(dsl_t.reshape(1, 1, npages * 32, 32), dtype=ttnn.bfloat16,
                              layout=ttnn.TILE_LAYOUT, device=dev)
        dsln = dsl_t.to(torch.float64).numpy().reshape(npages, 32, 32)

        smat = shift_matrices()
        s0m_np = np.concatenate([np.ones((1, 32, 32), dtype=np.float32), smat], axis=0)
        nmask = s0m_np.shape[0]
        s0m = ttnn.from_torch(torch.from_numpy(s0m_np.reshape(1, 1, nmask * 32, 32)).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        res["nmask"] = nmask

        wtens = ttnn.from_torch(torch.zeros(1, 1, 32 * WPAGES, 32).to(torch.bfloat16),
                                dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        vol = ttnn.from_torch(torch.zeros(1, 1, 32 * nb1 * ncore, 32).to(tdt), dtype=vol_fmt,
                              layout=ttnn.TILE_LAYOUT, device=dev)

        if not a.skip_parity:
            print("PARITY  (S0' stages the store, S2' reads what S0' wrote, S1' reads S2's W)",
                  flush=True)
            s0_parity(dev, res, dsl, s0m, sl, row_el, strip_tiles, dsln, npages_per_core, nmask,
                      a.shift_real, a.hoist, fid)
            # The S2' reference is what the DEVICE staged, so the chain is end to end.
            sln = ttnn.to_torch(sl)[0, 0].to(torch.float64).numpy()
            # deep enough that the S1' window at nstep 3 never wraps: nwin = 84 pages
            wpg = 128
            w1 = ttnn.from_torch(torch.zeros(1, 1, 32 * wpg, 32).to(torch.bfloat16),
                                 dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            for _nc in (a.chunk, 2 * a.chunk, 3 * a.chunk):
                s2_parity(dev, res, a.chunk, row_el, sl, sln, coef, cfn, selt, qn, offs_el, rho,
                          _nc, wpg)
            # S1' off the device's own W, at nstep = 1 and at the band's real depth
            rowidx1 = [(np.arange(NROWS) * NCOPY + rho).tolist()]
            pd = build_s2a(sl, coef, selt, w1, 1, 1, 1, row_el, offs_el * ELEM, rowidx1,
                           a.ncontrib, a.chunk)
            ttnn.generic_op([sl, coef, selt, w1], pd)
            ttnn.synchronize_device(dev)
            for ns in (1, 2, 3):
                s1_parity(dev, res, w1, m_t, ns, vol_fmt, vol_page, tdt, mkn,
                          "fp32dst" if dstacc else "bf16dst", wpg)
            ttnn.deallocate(w1)

        print(f"\nbox {a.box}: slice {2*a.box*a.box} B = {strips_per_slice} strips x "
              f"{strip_tiles} tiles;  L1 store {res['slice_l1_mb']:.1f} MB;  DRAM source "
              f"{res['dram_src_mb']:.0f} MB", flush=True)
        print(f"  {nslice:.0f} slices: S0' {nstrip} strips/core, S2' {a.nb2} W tiles/core x "
              f"{a.ncontrib} contributions ({nparticle/1e3:.0f}k particles), S1' {nb1} volume "
              f"tiles/core x {nwin}", flush=True)

        pd0 = build_s0(dsl, s0m, sl, NX, NY, nstrip, row_el, strip_tiles, npages_per_core, nmask,
                       a.shift_real, a.hoist, fid)
        pd2 = build_s2a(sl, coef, selt, wtens, NX, NY, a.nb2, row_el, offs_el * ELEM,
                        rowidx_of_core, a.ncontrib, a.chunk)
        pd1 = build_s1a(wtens, m_t, vol, NX, NY, nb1, a.nstep, vol_fmt, vol_page, WPAGES)
        ins0, ins2, ins1 = [dsl, s0m, sl], [sl, coef, selt, wtens], [wtens, m_t, vol]

        def arm():
            if not a.no_s0:
                ttnn.generic_op(ins0, pd0)   # DRAM slice -> replicated L1 store
            ttnn.generic_op(ins2, pd2)       # slice -> W, destination-stationary
            ttnn.generic_op(ins1, pd1)       # W -> volume, destination-stationary

        arm()
        ttnn.synchronize_device(dev)

        best, times, shas = float("inf"), [], []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            arm()
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            times.append(dt)
            best = min(best, dt)
            shas.append(hashlib.sha256(
                ttnn.to_torch(vol).to(torch.float32).numpy().tobytes()).hexdigest())

        # --- traffic, term by term, so every byte has a name -----------------------------------
        s0_dram_read = 0 if a.no_s0 else ncore * npages_per_core * TILE_B
        s0_l1_write = 0 if a.no_s0 else s0_dram_read * NCOPY
        s2_l1_read = ncore * a.nb2 * a.ncontrib * NROWS * WIN * ELEM
        s2_dram_write = ncore * a.nb2 * TILE_B
        s1_dram_read = ncore * (nwin + max(0, nb1 - 1) * a.nstep) * TILE_B
        s1_dram_write = ncore * nb1 * vol_page
        rate = nslice / best
        fl = floor_slices_s(a.box)
        traffic = {"s0_dram_read": s0_dram_read, "s0_l1_write": s0_l1_write,
                   "s2_l1_read": s2_l1_read, "s2_dram_write": s2_dram_write,
                   "s1_dram_read": s1_dram_read, "s1_dram_write": s1_dram_write}
        res.update({
            "wall_s": best, "all_wall_s": times, "sha256": shas[0], "sha_stable": len(set(shas)) == 1,
            "aa_delta_pct": 100 * (max(times) - min(times)) / min(times),
            "k_slices_per_s": rate / 1e3, "pct_of_floor": 100 * rate / fl,
            "ns_per_slice": best * 1e9 / nslice, "traffic_bytes": traffic,
            "per_slice_bytes": {k: v / nslice for k, v in traffic.items()},
            "implied_dram_read_gbs": (s0_dram_read + s1_dram_read) / best / 1e9,
            "implied_dram_write_gbs": (s2_dram_write + s1_dram_write) / best / 1e9,
            "implied_l1_gbs": (s0_l1_write + s2_l1_read) / best / 1e9})

        print(f"\nINTEGRATED BACKPROJECTION, box {a.box}, acc {a.acc}, "
              f"S0' {'ON' if not a.no_s0 else 'OFF'}{(' [' + a.tag + ']') if a.tag else ''}:")
        print(f"  wall {best*1e3:8.3f} ms for {nslice:.0f} slices  (runs "
              f"{', '.join(f'{t*1e3:.2f}' for t in times)} ms;  A/A spread "
              f"{res['aa_delta_pct']:.1f}%)")
        print(f"  {rate/1e3:9.1f} k slices/s   {100*rate/fl:5.2f}% of the {fl/1e6:.3f} M floor"
              f"   {best*1e9/nslice:8.1f} ns/slice")
        print(f"  DRAM read  {res['implied_dram_read_gbs']:6.1f} GB/s of a {READ_ROOF/1e9:.1f} roof"
              f"   ({res['per_slice_bytes']['s0_dram_read']:8.0f} B/slice slice + "
              f"{res['per_slice_bytes']['s1_dram_read']:6.0f} W)")
        print(f"  DRAM write {res['implied_dram_write_gbs']:6.1f} GB/s of a {WRITE_ROOF/1e9:.1f} roof"
              f"   ({res['per_slice_bytes']['s1_dram_write']:8.0f} B/slice volume + "
              f"{res['per_slice_bytes']['s2_dram_write']:6.0f} W)")
        print(f"  L1         {res['implied_l1_gbs']:6.1f} GB/s"
              f"   ({res['per_slice_bytes']['s0_l1_write']:8.0f} B/slice replication + "
              f"{res['per_slice_bytes']['s2_l1_read']:6.0f} assemblies)")
        print(f"  sha256 {shas[0]}  stable over {a.reps} reps: {res['sha_stable']}")
        suffix = f"_{a.tag}" if a.tag else ("" if not a.no_s0 else "_nos0")
        json.dump(res, open(HERE / f"bproj_e2e_box{a.box}_{a.acc}{suffix}.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
