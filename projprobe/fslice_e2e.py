#!/usr/bin/env python3
"""The integrated arm: stage 1 and stage 2 chained on device, one timed region, slices out.

Every projection rate before this one was `stage1_ns/slice/core + stage2_ns/slice/core`, from two
generic_op programs run on two different synthetic inputs, each turned into a per-slice figure by its
own amortisation constant (512/96 for stage 1, 25.13 for stage 2) and then added. That hides two
things an integrated run cannot hide: whether stage 1 can produce the layout stage 2 needs at all,
and whether the two stages agree on what a slice is.

They did not. Stage 1 counts BOTH complex components (512 W tiles per direction is 256 tiles per
plane x 2). Stage 2's timed kernel is real-valued end to end -- one scalar per output point, 2048 B
per 1024-point tile -- so the published rate priced half a slice for stage 2 and a whole one for
stage 1. This harness runs stage 2 `--components` times, so the default of 2 is a whole complex
slice through both stages.

CHAINING. Stage 2's reader takes per-row contiguous windows out of an L1-resident plane, so stage 1
has to hand it row-major rows, not packed tiles. That is what compute_zcollapse_rm.cpp does, in
32-row strips so the untilized block is a whole 512-wide padded plane row and the write stays bulk.
`pack_untilize_dest` cannot consume FPU-written DST (fslice_untilize.py mode 10: 1019/1024 wrong;
mode 8: bit-exact through copy_tile), so the strip is computed in two sweeps with a staging CB in
between to launder the DST arrangement.

The replication is a COST PROBE, as it was in section 28.2: 8 copies are written per strip so the
byte count and the addresses are the general shear's real ones, but copies 1..7 carry the unshifted
strip. The parity arm therefore runs a shear whose per-row offsets are multiples of 8, which reads
copy 0 only, and checks the whole chain -- stage 1's row-major W and stage 2's slice off it --
against an fp64 model.
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

# --- stage 1 CBs / stage 2 CBs (disjoint, the two programs never run at once) ---
CB_V, CB_MASK, CB_MID1, CB_OUT1, CB_MID2 = 0, 4, 8, 12, 16
CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, CB_MID, CB_INT, CB_INT2 = 0, 4, 8, 12, 16, 20, 24, 28

ELEM = 2
TILE_B = 32 * 32 * ELEM
BARRIER_EVERY = 4
NROWS, SRC_TILES = 32, 2
WIN = 32 * SRC_TILES
STRIP_TILES = 16          # set from the box in main(): a strip must be a whole plane row
NCOPY = 8
NPLANE = 28                     # S3's mean band over real HEALPix directions
PSI = 96                        # healpix order 3, the 3.75 deg step auto-refine converges to
NPAGES = 4096                   # volume pages the stage-1 reader cycles, DRAM resident
A = 1.31
# A half-space slice is pi*(N/2)^2/2 output points -- RELION's Friedel storage. 25,736 at box
# 256, and it scales with the box, so it cannot stay the constant the stage-2 harness used.
tiles_per_slice = lambda box: math.pi * (box / 2.0) ** 2 / 2.0 / 1024.0
# The floor, on roofs MEASURED on this card (projprobe/b0_roofs.json: read 406.6, write 173.7,
# rmw 297.3 GB/s) and the overlap the rmw arm implies, T = max(R, W) + 0.394*min(R, W).
# At box 256 the irreducible traffic is a 305,835 B band read (752.2 ns) and a 102,944 B slice
# write (592.6 ns), so the floor is 985.7 ns/slice. Both terms scale exactly as box^2.
# The old {256: 3.206e6, 384: 1.425e6, 512: 0.801e6} priced the slice WRITE with ttnn.add's
# 420.2 GB/s two-read-one-write number and was 2.24x too generous.
FLOOR_NS_256 = 985.7
FLOOR_READ_NS_256 = 752.2          # the read channel alone: the ceiling any lever can reach
floor_slices_s = lambda box, ns=FLOOR_NS_256: 1e9 / (ns * (box / 256.0) ** 2)
NX, NY = 13, 10


def plane_geom(box, pad=2):
    """Padded plane width in elements, and strips per direction counting both components."""
    row_el = pad * box
    tiles_per_plane = (row_el // 32) ** 2
    strip_tiles = row_el // 32
    strips_per_dir = 2 * tiles_per_plane // strip_tiles
    return row_el, strips_per_dir, strip_tiles


def sel_matrices(a, nsrc):
    p = [np.zeros((nsrc, 32), dtype=np.float32) for _ in range(3)]
    for u in range(32):
        k = int(math.floor(a * u))
        for d in range(3):
            if 0 <= k + d < nsrc:
                p[d][k + d, u] = 1.0
    return p


def cbset(cg, spec):
    out = []
    for i, page, depth in spec:
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        out.append(ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f]))
    return out


def mk(p, cg, ct, rt, cfg):
    return ttnn.KernelDescriptor(kernel_source=str(p),
                                 source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
                                 core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)


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


def build_stage1(v, m, w, nx, ny, nstrip, row_el, hoist, strip_of_core, shift_real, nmask):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])
    ncore = nx * ny
    rct = ([CB_V, CB_MASK, NPLANE, TILE_B, BARRIER_EVERY, NPLANE, nmask]
           + list(ttnn.TensorAccessorArgs(v).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(m).get_compile_time_args()))
    cct = [CB_V, CB_MASK, CB_MID1, CB_OUT1, NPLANE, NPLANE, NCOPY, STRIP_TILES, hoist,
           shift_real, CB_MID2]
    wct = ([CB_OUT1, row_el * ELEM, 32, NCOPY, STRIP_TILES, ncore]
           + list(ttnn.TensorAccessorArgs(w).get_compile_time_args()))
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            # Each core walks its own stretch of volume pages; nstrip*STRIP_TILES tiles of band
            # NPLANE each, so the reader never reuses a page inside a core.
            rrt[cx][cy] = [v.buffer_address(), m.buffer_address(), nstrip * STRIP_TILES,
                           (c * 37) % NPAGES, NPAGES]
            crt[cx][cy] = [nstrip]
            wrt[cx][cy] = [w.buffer_address(), nstrip, strip_of_core[c]]
            c += 1
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_zcollapse.cpp", cg, rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_zcollapse_rm.cpp", cg, cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(KDIR / "writer_strip.cpp", cg, wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbset(cg, [(CB_V, TILE_B, 2 * NPLANE), (CB_MASK, TILE_B, nmask),
                                     (CB_MID1, TILE_B, 2 * STRIP_TILES),
                                     (CB_OUT1, TILE_B, 2 * STRIP_TILES),
                                     (CB_MID2, TILE_B, 2)]))


def mode14_asm(nb, run_len, step_q):
    """Assemblies mode 14 consumes for `nb` output tiles, with the kernel's exact integer arithmetic.

    The reader has to issue precisely this many or the program deadlocks, so the count is derived the
    same way in both places: `acc` is the window start in Q16 mid-rows and `acc >> 21` its tile index,
    and a run needs the highest window start it reaches, plus the two tiles of the window itself.
    """
    tot, k0 = 0, 0
    while k0 < nb:
        t_run = min(run_len, nb - k0)
        acc, produced = 0, 0
        for _ in range(t_run):
            produced = max(produced, (acc >> 21) + 2)
            acc += step_q
        tot += produced
        k0 += t_run
    return tot


def build_stage2(w, sel, frac, out, nx, ny, nb, row_el, offs_bytes, rowidx_of_core,
                 mode=13, fid="HiFi4", cbtil=2, cbsrc=4, run_len=5, step_q=0):
    """`nb` is the number of OUTPUT TILES per core that mode 13 produces.

    The reader issues the SAME 2*nb assemblies whatever the mode, so a screen mode prices the
    same transaction stream as the built kernel and the two are comparable. What differs is what
    the compute engine does with them: mode 13 takes two per output tile, modes 5/6 one, and
    mode 0 only tilizes and packs (src_tiles out per assembly, hence the wider writer block).
    """
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])
    # Mode 14 is the one mode whose reader count is not 2*nb: that IS the lever. It produces the same
    # nb output tiles from fewer assemblies, so its sha256 stays comparable with mode 13's.
    asm = mode14_asm(nb, run_len, step_q) if mode == 14 else nb * 2
    nblk = nb if mode in (13, 14) else asm   # compute-loop trips
    tpb = SRC_TILES if mode == 0 else 1      # cb_out tiles the compute pushes per trip
    rct = ([CB_SRC, WIN * ELEM, NROWS, row_el * ELEM, SRC_TILES, BARRIER_EVERY, mode,
            CB_SEL, CB_FRAC, 3 * SRC_TILES, TILE_B, 3]
           + list(ttnn.TensorAccessorArgs(w).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(sel).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(frac).get_compile_time_args()))
    cct = [CB_SRC, CB_TIL, CB_SEL, CB_FRAC, CB_OUT, SRC_TILES, mode, CB_MID, CB_INT, CB_INT2]
    wct = [CB_OUT, TILE_B, tpb] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = ([w.buffer_address(), asm, 0] + [int(o) for o in offs_bytes]
                           + [sel.buffer_address(), frac.buffer_address()]
                           + [int(i) for i in rowidx_of_core[c]])
            crt[cx][cy] = [nblk, run_len, step_q]
            wrt[cx][cy] = [out.buffer_address(), nblk, c * nblk * tpb]
            c += 1
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_fslice.cpp", cg, rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_fslice.cpp", cg, cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=getattr(ttnn.MathFidelity, fid))),
        mk(KDIR / "writer_fslice.cpp", cg, wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=cbset(cg, [(CB_SRC, TILE_B, cbsrc * BARRIER_EVERY * SRC_TILES),
                                     (CB_TIL, TILE_B, cbtil * SRC_TILES), (CB_SEL, TILE_B, 3 * SRC_TILES),
                                     (CB_FRAC, TILE_B, 4), (CB_OUT, TILE_B, 4), (CB_MID, TILE_B, 2),
                                     (CB_INT, TILE_B, 4), (CB_INT2, TILE_B, 4)]))


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


def frac_tiles(h):
    f5 = np.zeros((3, 32, 32), dtype=np.float32)
    for r in range(NROWS):
        for u in range(32):
            w = h[r] + (A * u - math.floor(A * u))
            m = math.floor(w)
            wq = w - m
            f5[0, r, u] = (1 - wq) * (1 - m)
            f5[1, r, u] = (1 - wq) * m + wq * (1 - m)
            f5[2, r, u] = wq * m
    return f5


def parity(dev, res, row_el, mkn, vol_t, voln, m_t, shift_real, nmask, bs, cs):
    """One core, one strip: does the chain compute the right thing end to end?"""
    w = ttnn.from_torch(torch.zeros(1, 1, 32 * NCOPY, row_el).to(torch.bfloat16),
                        dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                        memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED,
                                                        ttnn.BufferType.L1))
    pd1 = build_stage1(vol_t, m_t, w, 1, 1, 1, row_el, 0, [0], shift_real, nmask)
    ttnn.generic_op([vol_t, m_t, w], pd1)
    ttnn.synchronize_device(dev)
    got = ttnn.to_torch(w).reshape(32 * NCOPY, row_el).to(torch.float64).numpy()
    ref = np.zeros((32, row_el))
    for t in range(STRIP_TILES):
        pages = [(0 + t * NPLANE + p) % NPAGES for p in range(NPLANE)]
        ref[:, 32 * t:32 * t + 32] = np.einsum("pxy,pxy->xy", mkn, voln[pages])
    g0 = got[0::NCOPY][:32]
    rel1 = float(np.linalg.norm(g0 - ref) / max(np.linalg.norm(ref), 1e-300))
    print(f"  stage 1 -> row-major W strip vs fp64: rel L2 {rel1:.4e}   "
          f"max|diff| {np.abs(g0 - ref).max():.3e}", flush=True)
    res["parity_stage1_rel_l2"] = rel1

    # With the cost-probe replication only copy 0 is real, so the chain can only be checked under a
    # restricted shear (every per-row offset a multiple of 8, rho == 0). With the real shifts on, the
    # GENERAL shear is checkable -- and that is the configuration the timed arm actually runs.
    k0, h, rho, offs_el = shear(bs, cs)
    if not shift_real:
        assert (rho == 0).all()
    sel_np = sel_matrices(A, WIN)
    selt = np.concatenate([sel_np[d].reshape(SRC_TILES, 32, 32).reshape(-1, 32) for d in range(3)], 0)
    sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                          dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    frac = ttnn.from_torch(torch.from_numpy(frac_tiles(h).reshape(1, 1, 96, 32)).to(torch.bfloat16),
                           dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    out = ttnn.from_torch(torch.zeros(1, 1, 32, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=dev)
    rowidx = [(np.arange(NROWS) * NCOPY + rho).tolist()]
    pd2 = build_stage2(w, sel, frac, out, 1, 1, 1, row_el, offs_el * ELEM, rowidx)
    ttnn.generic_op([w, sel, frac, out], pd2)
    ttnn.synchronize_device(dev)
    g = ttnn.to_torch(out).reshape(32, 32).to(torch.float64).numpy()

    # fp64 model of the fused two-pass, on the W the DEVICE actually produced.
    wn = torch.from_numpy(g0).to(torch.bfloat16).to(torch.float64).numpy()

    def p1(win):
        o = np.zeros((32, 32))
        for r in range(NROWS):
            for u in range(32):
                q = A * u + h[r]
                j = math.floor(q)
                f = q - j
                o[r, u] = (1 - f) * win[r, j] + f * win[r, j + 1]
        return o

    # Row r reads copy rho[r] at element offset offs_el[r], and copy q is the strip shifted left by
    # q, so the window starts at offs_el[r] + rho[r] = k0[r] of the unshifted strip either way.
    w0 = np.stack([wn[r, k0[r]:k0[r] + WIN] for r in range(NROWS)])
    it = np.concatenate([p1(w0), p1(w0)], axis=0).T
    ref2 = np.zeros((32, 32))
    for r in range(32):
        for u in range(32):
            q = A * u + h[r]
            j = math.floor(q)
            f = q - j
            ref2[r, u] = (1 - f) * it[r, j] + f * it[r, j + 1]
    rel2 = float(np.linalg.norm(g - ref2) / max(np.linalg.norm(ref2), 1e-300))
    print(f"  chained stage 1 -> stage 2 slice tile vs fp64: rel L2 {rel2:.4e}   "
          f"max|diff| {np.abs(g - ref2).max():.3e}", flush=True)
    res["parity_chain_rel_l2"] = rel2
    for t_ in (w, sel, frac, out):
        ttnn.deallocate(t_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("box", type=int, nargs="?", default=256)
    ap.add_argument("--nstrip", type=int, default=10)
    ap.add_argument("--components", type=int, default=2)
    ap.add_argument("--hoist", type=int, default=0)
    ap.add_argument("--shift", type=int, default=1,
                    help="1 = the replicated copies carry their real shifts; 0 = section 28.2's cost probe")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--skip-parity", action="store_true")
    ap.add_argument("--mode", type=int, default=13, help="stage-2 compute mode; 13 is the built kernel")
    ap.add_argument("--fid", default="HiFi4", choices=["LoFi", "HiFi2", "HiFi3", "HiFi4"],
                    help="stage-2 math fidelity (lever F)")
    ap.add_argument("--cbtil", type=int, default=2, help="cb_til depth in units of SRC_TILES (lever O)")
    ap.add_argument("--cbsrc", type=int, default=4,
                    help="cb_src depth in units of BARRIER_EVERY*SRC_TILES (lever O)")
    ap.add_argument("--runlen", type=int, default=5,
                    help="output tiles per tile-column run (lever B); the disc gives 5.03")
    ap.add_argument("--dscale", type=float, default=1.1228,
                    help="D = 1/cos(theta) for lever B; s4_geom puts the mean at 1.1228")
    ap.add_argument("--split", action="store_true",
                    help="also time the two stages separately, after the chained arm")
    a = ap.parse_args()

    global STRIP_TILES
    row_el, strips_per_dir, STRIP_TILES = plane_geom(a.box)
    ncore = NX * NY
    dev = ttnn.open_device(device_id=0)
    res = {"box": a.box, "row_el": row_el, "strips_per_dir": strips_per_dir, "nplane": NPLANE,
           "psi": PSI, "ncopy": NCOPY, "nstrip": a.nstrip, "components": a.components,
           "hoist_init": a.hoist, "ncore": ncore, "mode": a.mode, "fid": a.fid,
           "cbtil": a.cbtil, "cbsrc": a.cbsrc, "run_len": a.runlen, "dscale": a.dscale}
    try:
        rng = np.random.default_rng(61)
        vol = rng.integers(-100, 100, size=(NPAGES * 32, 32)).astype(np.float32)
        vt = torch.from_numpy(vol).to(torch.bfloat16)
        voln = vt.to(torch.float64).numpy().reshape(NPAGES, 32, 32)
        vol_t = ttnn.from_torch(vt.reshape(1, 1, NPAGES * 32, 32), dtype=ttnn.bfloat16,
                                layout=ttnn.TILE_LAYOUT, device=dev)
        mkf = masks()
        mkn = torch.from_numpy(mkf).to(torch.bfloat16).to(torch.float64).numpy()
        # The shift matrices ride in on the mask tensor's one-off load.
        mstack = np.concatenate([mkf, shift_matrices()], 0) if a.shift else mkf
        nmask = mstack.shape[0]
        m_t = ttnn.from_torch(torch.from_numpy(mstack.reshape(1, 1, nmask * 32, 32)).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        res["nmask"] = nmask
        res["shift_real"] = a.shift

        if not a.skip_parity:
            print("PARITY", flush=True)
            parity(dev, res, row_el, mkn, vol_t, voln, m_t, a.shift, nmask,
                   *((0.77, 5.3) if a.shift else (8.0, 0.0)))

        # --- the integrated arm ---
        # W holds one strip per core. A batch longer than that wraps, which keeps the addresses and
        # the byte count real while bounding L1 to what one block of directions actually needs.
        w_rows = ncore * 32 * NCOPY
        print(f"\nW buffer: {ncore} strips x 32 rows x {NCOPY} copies x {row_el} el = "
              f"{w_rows * row_el * ELEM / 2**20:.1f} MB L1", flush=True)
        w = ttnn.from_torch(torch.zeros(1, 1, w_rows, row_el).to(torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                            memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED,
                                                            ttnn.BufferType.L1))
        # slices produced: nstrip*ncore*STRIP_TILES W tiles / (strips_per_dir*STRIP_TILES) per
        # direction, x PSI in-plane rotations.
        w_tiles = a.nstrip * ncore * STRIP_TILES
        ndir = w_tiles / (strips_per_dir * STRIP_TILES)
        nslice = ndir * PSI
        nb = int(round(nslice * tiles_per_slice(a.box) * a.components / ncore))
        res.update({"w_mb": w_rows * row_el * ELEM / 2**20, "ndir": ndir, "nslice": nslice, "nb": nb})
        print(f"batch: {ndir:.1f} directions x {PSI} psi = {nslice:.0f} slices; "
              f"stage 1 {a.nstrip} strips/core, stage 2 {nb} tiles/core", flush=True)

        k0, h, rho, offs_el = shear(0.77, 5.3)
        assert len(set(rho.tolist())) == NCOPY, sorted(set(rho.tolist()))
        sel_np = sel_matrices(A, WIN)
        selt = np.concatenate([sel_np[d].reshape(SRC_TILES, 32, 32).reshape(-1, 32)
                               for d in range(3)], 0)
        sel = ttnn.from_torch(torch.from_numpy(selt).to(torch.bfloat16).reshape(1, 1, -1, 32),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        frac = ttnn.from_torch(torch.from_numpy(frac_tiles(h).reshape(1, 1, 96, 32)).to(torch.bfloat16),
                               dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        # A screen mode produces a different number of output tiles per core from the same reader
        # stream, so the destination has to be sized for the mode or the writer walks off the end.
        step_q = int(round(32.0 * a.dscale * 65536))
        nblk = nb if a.mode in (13, 14) else nb * 2
        tpb = SRC_TILES if a.mode == 0 else 1
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nblk * tpb * ncore, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        strip_of_core = list(range(ncore))
        rowidx_of_core = [((c * 32 + np.arange(NROWS)) * NCOPY + rho).tolist() for c in range(ncore)]
        pd1 = build_stage1(vol_t, m_t, w, NX, NY, a.nstrip, row_el, a.hoist, strip_of_core,
                           a.shift, nmask)
        pd2 = build_stage2(w, sel, frac, out, NX, NY, nb, row_el, offs_el * ELEM, rowidx_of_core,
                           a.mode, a.fid, a.cbtil, a.cbsrc, a.runlen, step_q)
        if a.mode == 14:
            asm14 = mode14_asm(nb, a.runlen, step_q)
            res.update({"asm": asm14, "asm_ratio": asm14 / (2 * nb), "step_q": step_q})
            print(f"lever B: D = {a.dscale}, run {a.runlen} -> {asm14} assemblies/core against "
                  f"{2 * nb}, r = {asm14 / (2 * nb):.4f}", flush=True)

        ins1, ins2 = [vol_t, m_t, w], [w, sel, frac, out]
        ttnn.generic_op(ins1, pd1)
        ttnn.generic_op(ins2, pd2)
        ttnn.synchronize_device(dev)

        best, times = float("inf"), []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            ttnn.generic_op(ins1, pd1)          # stage 1: volume -> replicated row-major W
            ttnn.generic_op(ins2, pd2)          # stage 2: W -> complex slices, no host in between
            ttnn.synchronize_device(dev)
            dt = time.perf_counter() - t0
            times.append(dt)
            best = min(best, dt)
        if a.split:
            # The two stages are separate programs and cannot overlap, so timing them apart adds one
            # device sync (~50 us against a 9 ms arm) and does not change what either one costs.
            def stage_ns(ins, pd):
                ttnn.generic_op(ins, pd); ttnn.synchronize_device(dev)
                bs = float("inf")
                for _ in range(a.reps):
                    t0 = time.perf_counter(); ttnn.generic_op(ins, pd)
                    ttnn.synchronize_device(dev); bs = min(bs, time.perf_counter() - t0)
                return bs * 1e9 / nslice
            s1_ns, s2_ns = stage_ns(ins1, pd1), stage_ns(ins2, pd2)
            res.update({"stage1_ns_per_slice": s1_ns, "stage2_ns_per_slice": s2_ns})
            print(f"  split: stage 1 {s1_ns:7.1f} + stage 2 {s2_ns:7.1f} = {s1_ns + s2_ns:7.1f} "
                  f"ns/slice (chained {best * 1e9 / nslice:.1f})", flush=True)

        sha = hashlib.sha256(ttnn.to_torch(out).view(torch.int16).numpy().tobytes()).hexdigest()
        rate = nslice / best
        floor = floor_slices_s(a.box)
        res.update({"wall_s": best, "all_wall_s": times, "sha256": sha,
                    "k_slices_per_s": rate / 1e3, "pct_of_floor": 100 * rate / floor,
                    "ns_per_slice_per_core": best * 1e9 / nslice})
        print(f"\nINTEGRATED, box {a.box}, {a.components} component(s), general shear, shifts {'REAL' if a.shift else 'COST PROBE'}:")
        print(f"  wall {best * 1e3:8.3f} ms for {nslice:.0f} slices   (runs "
              f"{', '.join(f'{t*1e3:.2f}' for t in times)} ms)")
        print(f"  {rate / 1e3:9.1f} k slices/s   {100 * rate / floor:5.2f}% of the "
              f"{floor / 1e6:.4f} M measured-overlap floor   {best * 1e9 / nslice:8.1f} ns/slice/core")
        print(f"  sha256 {sha}")
        tag = (f"fslice_e2e_box{a.box}_c{a.components}_h{a.hoist}_s{a.shift}"
               + (f"_m{a.mode}" if a.mode != 13 else "")
               + (f"_{a.fid}" if a.fid != "HiFi4" else "")
               + (f"_t{a.cbtil}" if a.cbtil != 2 else "") + (f"_r{a.cbsrc}" if a.cbsrc != 4 else "")
               + (f"_L{a.runlen}D{a.dscale}" if a.mode == 14 else ""))
        json.dump(res, open(HERE / f"{tag}.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
