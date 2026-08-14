#!/usr/bin/env python3
"""The integrated adjoint arm: stage 2' and stage 1' chained on device, one timed region.

Backprojection run DESTINATION-STATIONARY, which is what makes the no-scatter claim pay:

  S2'  slice -> W    W'[r, j] = sum_d ( c_d(r, u) * y[r, u] ) . P_d^T[u, j]
                     Destination = one ALIGNED W tile. Its contributions are 32 rows of the slice,
                     each at its own offset, which is the forward's own reader unchanged
                     (reader_fslice.cpp) pointed at an L1-resident slice instead of an L1-resident
                     plane. The write is one bulk page.
  S1'  W -> volume   V'[X, Y, z] = sum_{d: band covers z} mask[z - z0_d][X, Y] * W_d[X, Y]
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

AMORTISATION, every constant stated rather than assumed.
  n_orient = 768 HEALPix order-3 directions x 96 psi = 73,728 slices per iteration.
  S2'      one aligned W tile per block, `ncontrib` psi accumulated into it. The (W tile, psi)
           incidence count per slice is the slice's own tile count, pi*(box/2)^2/2/1024 per
           component x 2 components, because the shear preserves area.
  S1'      one volume tile per block. The padded half-space accumulator is (2box)^2 x (box+1)
           voxels x 3 components (complex data + REAL WEIGHT), written once per iteration, so
           3*(2box)^2*(box+1)/1024/73,728 tile writes per slice. Running nwin = 28 contributions on
           every one of those blocks OVER-prices S1' by 1.7x against the true (direction, volume
           tile) incidence count, because the padded corners carry a write but no contribution. The
           measured rate is therefore a lower bound, deliberately.

The accumulation carry is fp32 by construction: one acquire can only matmul what is already staged
in L1, so contributions are chunked and the running sum crosses cb_acc between chunks. `--acc bf16`
runs the identical program with that carry in bf16, which is the device-side half of the
accumulation-precision answer (the host model over random contributions is the other half).
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

# S2' CBs / S1' CBs (disjoint programs, never resident at once)
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


def build_s2a(sl, coef, selt, w, nx, ny, nb, row_el, offs_bytes, rowidx_of_core, ncontrib,
              chunk, fmt, page):
    """S2': the forward's stage-2 reader and writer, UNCHANGED, around the adjoint's compute."""
    cg = grid(nx, ny)
    nselt = 3 * SRC_TILES
    rct = ([CB_SRC, WIN * ELEM, NROWS, row_el * ELEM, SRC_TILES, BARRIER_EVERY, 13,
            CB_SELT, CB_COEF, nselt, TILE_B, 3]
           + list(ttnn.TensorAccessorArgs(sl).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(selt).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(coef).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_COEF, CB_SELT, CB_MID, CB_ACC, CB_OUT, SRC_TILES, ncontrib, chunk]
    wct = [CB_OUT, page, 1] + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
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
        mk(KDIR / "compute_bproj_ds.cpp", cg, cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                        fp32_dest_acc_en=(fmt == ttnn.float32))),
        mk(KDIR / "writer_fslice.cpp", cg, wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs(cg, [
        (CB_SRC, ttnn.bfloat16, TILE_B, 4 * BARRIER_EVERY * SRC_TILES),
        # Every CB this kernel PACKS into carries one format. The tilize reconfigures the packer
        # to cb_til, and a later pack_tile to a differently-formatted CB is not reconfigured back:
        # the first build wrote fp32 DST as bf16 into an fp32 buffer and parity came back 1.2e31.
        (CB_TIL, ttnn.bfloat16, TILE_B, 2 * SRC_TILES),
        (CB_COEF, ttnn.bfloat16, TILE_B, 3),
        (CB_SELT, ttnn.bfloat16, TILE_B, nselt),
        (CB_MID, ttnn.bfloat16, TILE_B, 2 * 3 * SRC_TILES * chunk),
        (CB_ACC, fmt, page, 2),
        (CB_OUT, fmt, page, 2)]))


def build_s1a(w, m, vol, nx, ny, nb, nstep, fmt, page):
    """S1': destination-stationary z-spread, sliding window over the contributing directions."""
    cg = grid(nx, ny)
    nwin = NPLANE * nstep
    rct = ([CB_W, CB_MASK, NPLANE, nstep, page, BARRIER_EVERY]
           + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(m).get_compile_time_args()))
    cct = [CB_W, CB_MASK, CB_VOUT, NPLANE, nstep]
    wct = [CB_VOUT, page, 1] + list(ttnn.TensorAccessorArgs(vol).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = [w.buffer_address(), m.buffer_address(), nb, (c * 29) % WPAGES, WPAGES]
            crt[cx][cy] = [nb]
            wrt[cx][cy] = [vol.buffer_address(), nb, c * nb]
            c += 1
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_zspread_acc.cpp", cg, rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_zspread_acc.cpp", cg, cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4,
                                        fp32_dest_acc_en=(fmt == ttnn.float32))),
        mk(KDIR / "writer_fslice.cpp", cg, wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbs(cg, [
        (CB_W, fmt, page, nwin + nstep * BARRIER_EVERY * 2),
        (CB_MASK, fmt, page, NPLANE),
        (CB_VOUT, fmt, page, 2)]))


def s2_reference(sln, cfn, qn, rho, offs_el, ncontrib):
    """fp64 adjoint of exactly what the reader assembled, accumulated over ncontrib contributions."""
    win = np.stack([sln[r * NCOPY + rho[r], offs_el[r]:offs_el[r] + WIN] for r in range(NROWS)])
    one = np.zeros((32, 32))
    for d in range(3):
        one += (cfn[d] * win) @ qn[d]
    return one * ncontrib, win


def s2_parity(dev, res, a, row_el, sl, sln, coef, cfn, selt, qn, offs_el, rho, fmt, page, tdt, nc2):
    """One core, one W tile, nc2 contributions -- run at 1, 2 and 3 chunks so the cb_acc carry is
    isolated from the arithmetic. A scale-only residual means contributions are being LOST rather
    than computed wrong, and the least-squares scale says how many of them landed."""
    w1 = ttnn.from_torch(torch.zeros(1, 1, 32 * WPAGES, 32).to(tdt), dtype=fmt,
                         layout=ttnn.TILE_LAYOUT, device=dev)
    rowidx1 = [(np.arange(NROWS) * NCOPY + rho).tolist()]
    pd = build_s2a(sl, coef, selt, w1, 1, 1, 1, row_el, offs_el * ELEM, rowidx1, nc2,
                   a.chunk, fmt, page)
    ttnn.generic_op([sl, coef, selt, w1], pd)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(w1)[0, 0, :32, :].to(torch.float64).numpy()
    ref, _ = s2_reference(sln, cfn, qn, rho, offs_el, nc2)
    one = ref / nc2
    rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300))
    k = float((got * one).sum() / (one * one).sum())
    resid = float(np.linalg.norm(got - k * one) / max(np.linalg.norm(got), 1e-300))
    err = np.abs(got - ref).ravel()
    print("  S2' nc2 %3d (%d chunk): rel L2 %.4e   landed %7.3f of %d   "
          "scale-removed residual %.4e" % (nc2, nc2 // a.chunk, rel, k, nc2, resid), flush=True)
    res["parity_s2_nc%d" % nc2] = {"rel_l2": rel, "landed": k, "resid": resid,
                                   "err_pct": {q: float(np.percentile(err, q))
                                               for q in (50, 90, 99, 100)}}
    ttnn.deallocate(w1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("box", type=int, nargs="?", default=256)
    ap.add_argument("--ncontrib", type=int, default=48)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--nstep", type=int, default=1)
    ap.add_argument("--nb2", type=int, default=8)
    ap.add_argument("--acc", choices=["fp32", "bf16"], default="fp32")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-parity", action="store_true")
    a = ap.parse_args()

    assert a.ncontrib % a.chunk == 0
    fmt = ttnn.float32 if a.acc == "fp32" else ttnn.bfloat16
    page = F32_TILE_B if a.acc == "fp32" else TILE_B
    tdt = torch.float32 if a.acc == "fp32" else torch.bfloat16
    row_el = 2 * a.box
    ncore = NX * NY
    nwin = NPLANE * a.nstep

    dev = ttnn.open_device(device_id=0)
    res = {"box": a.box, "row_el": row_el, "ncontrib": a.ncontrib, "chunk": a.chunk,
           "nstep": a.nstep, "nwin": nwin, "acc": a.acc, "ncore": ncore, "src_tiles": SRC_TILES,
           "read_roof_gbs": READ_ROOF / 1e9, "write_roof_gbs": WRITE_ROOF / 1e9,
           "floor_slices_s": floor_slices_s(a.box)}
    try:
        rng = np.random.default_rng(97)
        sl_rows = ncore * 32 * NCOPY
        sl_np = rng.integers(-100, 100, size=(sl_rows, row_el)).astype(np.float32)
        sl_t = torch.from_numpy(sl_np).to(torch.bfloat16)
        sln = sl_t.to(torch.float64).numpy()
        sl = ttnn.from_torch(sl_t.reshape(1, 1, sl_rows, row_el), dtype=ttnn.bfloat16,
                             layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
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
        m_t = ttnn.from_torch(torch.from_numpy(mkf.reshape(1, 1, NPLANE * 32, 32)).to(tdt),
                              dtype=fmt, layout=ttnn.TILE_LAYOUT, device=dev)
        mkn = torch.from_numpy(mkf).to(tdt).to(torch.float64).numpy()

        rowidx_of_core = [((c * 32 + np.arange(NROWS)) * NCOPY + rho).tolist() for c in range(ncore)]

        inc_per_slice = tiles_per_slice(a.box) * 2
        nslice = a.nb2 * ncore * a.ncontrib / inc_per_slice
        nb1 = max(1, int(round(nslice * vol_tiles_per_slice(a.box) / ncore)))
        res.update({"nb2": a.nb2, "nb1": nb1, "nslice": nslice, "inc_per_slice": inc_per_slice,
                    "vol_tiles_per_slice": vol_tiles_per_slice(a.box)})

        wtens = ttnn.from_torch(torch.zeros(1, 1, 32 * WPAGES, 32).to(tdt), dtype=fmt,
                                layout=ttnn.TILE_LAYOUT, device=dev)
        vol = ttnn.from_torch(torch.zeros(1, 1, 32 * nb1 * ncore, 32).to(tdt), dtype=fmt,
                              layout=ttnn.TILE_LAYOUT, device=dev)

        if not a.skip_parity:
            print("PARITY", flush=True)
            # --- S2': one core, one W tile, 2*chunk contributions so the cb_acc carry is live ---
            for _nc in (a.chunk, 2 * a.chunk, 3 * a.chunk):
                s2_parity(dev, res, a, row_el, sl, sln, coef, cfn, selt, qn, offs_el, rho,
                          fmt, page, tdt, _nc)
            nc2 = 2 * a.chunk
            w1 = ttnn.from_torch(torch.zeros(1, 1, 32 * WPAGES, 32).to(tdt), dtype=fmt,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
            rowidx1 = [(np.arange(NROWS) * NCOPY + rho).tolist()]
            pd = build_s2a(sl, coef, selt, w1, 1, 1, 1, row_el, offs_el * ELEM, rowidx1, nc2,
                           a.chunk, fmt, page)
            ttnn.generic_op([sl, coef, selt, w1], pd)
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(w1)[0, 0, :32, :].to(torch.float64).numpy()
            ref, win = s2_reference(sln, cfn, qn, rho, offs_el, nc2)
            rel = float(np.linalg.norm(got - ref) / max(np.linalg.norm(ref), 1e-300))
            print(f"    |got| {np.linalg.norm(got):.4e}  |ref| {np.linalg.norm(ref):.4e}", flush=True)
            # One device run, many hypotheses: which arrangement of the operands does the kernel
            # actually compute? Cheaper than a device round trip per guess.
            cand = {}
            cand["as-built"] = ref
            cand["q^T"] = sum((cfn[d] * win) @ qn[d].T for d in range(3)) * nc2
            cand["win^T"] = sum((cfn[d] * win.T) @ qn[d] for d in range(3)) * nc2
            cand["c^T"] = sum((cfn[d].T * win) @ qn[d] for d in range(3)) * nc2
            cand["no-coef"] = sum(win @ qn[d] for d in range(3)) * nc2
            cand["coef-after"] = sum(cfn[d] * (win @ qn[d]) for d in range(3)) * nc2
            cand["q^T win^T"] = sum((cfn[d] * win.T) @ qn[d].T for d in range(3)) * nc2
            cand["out^T"] = ref.T
            for k, v in sorted(cand.items(), key=lambda kv: np.linalg.norm(got - kv[1])):
                print(f"      {k:14s} rel L2 {np.linalg.norm(got-v)/max(np.linalg.norm(v),1e-30):.4e}",
                      flush=True)
            err = np.abs(got - ref).ravel()
            print(f"  S2' slice -> W tile vs fp64, {nc2} contributions: rel L2 {rel:.4e}  "
                  f"max {err.max():.3e}  p50 {np.percentile(err,50):.3e}  "
                  f"p99 {np.percentile(err,99):.3e}", flush=True)
            res["parity_s2_rel_l2"] = rel
            res["parity_s2_err_pct"] = {p: float(np.percentile(err, p)) for p in (50, 90, 99, 100)}
            res["parity_s2_ref_rms"] = float(np.sqrt((ref ** 2).mean()))

            # --- S1': one core, one volume tile, nwin contributions off the DEVICE's own W ---
            v1 = ttnn.from_torch(torch.zeros(1, 1, 32 * 4, 32).to(tdt), dtype=fmt,
                                 layout=ttnn.TILE_LAYOUT, device=dev)
            pd1p = build_s1a(w1, m_t, v1, 1, 1, 1, a.nstep, fmt, page)
            ttnn.generic_op([w1, m_t, v1], pd1p)
            ttnn.synchronize_device(dev)
            gv = ttnn.to_torch(v1)[0, 0, :32, :].to(torch.float64).numpy()
            wn = ttnn.to_torch(w1).reshape(WPAGES, 32, 32).to(torch.float64).numpy()
            refv = np.zeros((32, 32))
            for i in range(nwin):
                refv += wn[i % WPAGES] * mkn[(nwin - 1 - i) // a.nstep]
            relv = float(np.linalg.norm(gv - refv) / max(np.linalg.norm(refv), 1e-300))
            ev = np.abs(gv - refv).ravel()
            print(f"  S1' W -> volume tile vs fp64, {nwin} contributions: rel L2 {relv:.4e}  "
                  f"max {ev.max():.3e}  p99 {np.percentile(ev,99):.3e}", flush=True)
            res["parity_s1_rel_l2"] = relv
            res["parity_s1_err_pct"] = {p: float(np.percentile(ev, p)) for p in (50, 90, 99, 100)}
            for t_ in (w1, v1):
                ttnn.deallocate(t_)

        print(f"\nslice store {res['slice_l1_mb']:.1f} MB L1;  {nslice:.0f} slices; "
              f"S2' {a.nb2} W tiles/core x {a.ncontrib} psi;  S1' {nb1} volume tiles/core "
              f"x {nwin} contributions", flush=True)

        pd2 = build_s2a(sl, coef, selt, wtens, NX, NY, a.nb2, row_el, offs_el * ELEM,
                        rowidx_of_core, a.ncontrib, a.chunk, fmt, page)
        pd1 = build_s1a(wtens, m_t, vol, NX, NY, nb1, a.nstep, fmt, page)
        ins2, ins1 = [sl, coef, selt, wtens], [wtens, m_t, vol]

        ttnn.generic_op(ins2, pd2)
        ttnn.generic_op(ins1, pd1)
        ttnn.synchronize_device(dev)

        best, times = float("inf"), []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            ttnn.generic_op(ins2, pd2)      # slice -> W, destination-stationary
            ttnn.generic_op(ins1, pd1)      # W -> volume, destination-stationary
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            times.append(dt)
            best = min(best, dt)
        sha = hashlib.sha256(ttnn.to_torch(vol).to(torch.float32).numpy().tobytes()).hexdigest()
        rate = nslice / best
        fl = floor_slices_s(a.box)
        # bytes/time against the roofs, in both directions -- the standing sanity check.
        rb = 2 * a.box * a.box + WIN * ELEM * 0
        res.update({"wall_s": best, "all_wall_s": times, "sha256": sha,
                    "k_slices_per_s": rate / 1e3, "pct_of_floor": 100 * rate / fl,
                    "ns_per_slice_per_core": best * 1e9 / nslice,
                    "implied_read_gbs": rate * rb / 1e9,
                    "implied_write_gbs": rate * vol_bytes(a.box) / NORIENT / 1e9})
        print(f"\nINTEGRATED BACKPROJECTION, box {a.box}, acc {a.acc}:")
        print(f"  wall {best*1e3:8.3f} ms for {nslice:.0f} slices  (runs "
              f"{', '.join(f'{t*1e3:.2f}' for t in times)} ms)")
        print(f"  {rate/1e3:9.1f} k slices/s   {100*rate/fl:5.2f}% of the {fl/1e6:.3f} M floor"
              f"   {best*1e9/nslice:8.1f} ns/slice/core")
        print(f"  implied slice read {res['implied_read_gbs']:6.1f} GB/s of a {READ_ROOF/1e9:.1f} "
              f"roof;  volume write {res['implied_write_gbs']:6.2f} GB/s of a "
              f"{WRITE_ROOF/1e9:.1f} roof")
        print(f"  sha256 {sha}")
        json.dump(res, open(HERE / f"bproj_e2e_box{a.box}_{a.acc}.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
