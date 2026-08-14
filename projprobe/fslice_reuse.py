#!/usr/bin/env python3
"""Plane reuse in the z-collapse: read each plane once and let it serve several output tiles.

Section 25.1 measured the z-collapse at 97-105% reads -- the multiplies are free -- and section 25.2
refuted the sub-blocking lever on that basis. What is left is read amplification: a tile fetches 28
planes (57 KB) to produce 2 KB, and only about 2 of the 28 carry information.

Adjacent output tiles share most of their band, so walking them in an order that keeps the band centre
nearly constant turns those reads into reuse. The band centre of tile (TX, TY) is 32(a*TX + b*TY), and
for the measured a = 0.43, b = 0.29 a +X step shifts it 13.76 planes, +Y 9.28, and the diagonal (1, -1)
only 4.48 -- so the diagonal walk should read about 4.5 planes per output tile instead of 28.

`shift` is planes entering the window per output tile. The circular buffer is the window: the compute
pops `shift` from the front and the reader pushes `shift` at the back. With an INTEGER shift the mask set
is identical for every tile, because the window start and the band centre advance together, so this is
correct and not just a cost probe -- verified against fp64 at every shift.

shift == nplane reproduces the old no-reuse behaviour, which is the baseline the speedups are against.
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
CB_V, CB_MASK, CB_OUT = 0, 8, 16
ELEM = 2
TILE_B = 32 * 32 * ELEM
BARRIER_EVERY = 4
NB = 200
NPAGES = 512
NPLANE = 28                    # the S3 mean band
# Shifts must DIVIDE nplane. Every push into the plane window is `shift` tiles, and a circular
# buffer whose size is not a multiple of the push size splits a push across the wrap, which hung
# the first run of this screen. 7 stands in for the diagonal walk (4.48) and 14 for +X (13.76).
SHIFTS = (28, 14, 7, 4, 2, 1)
PSI_PER_DIRECTION = 96
TILES_PER_SLICE = 25736 / 1024.0
TILES_PER_DIRECTION = (512 // 32) ** 2 * 2
FLOOR_SLICES_S = 3.20e6
STAGE2_NS_PER_SLICE_PER_CORE = 782.1     # measured, section 23


def build(dev, v, m, out, nx, ny, nb, shift, ncopy=1):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    rct = ([CB_V, CB_MASK, NPLANE, TILE_B, BARRIER_EVERY, shift]
           + list(ttnn.TensorAccessorArgs(v).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(m).get_compile_time_args()))
    cct = [CB_V, CB_MASK, CB_OUT, NPLANE, NPLANE, shift, ncopy]
    wct = [CB_OUT, TILE_B, ncopy] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(ny):
        for cx in range(nx):
            rrt[cx][cy] = [v.buffer_address(), m.buffer_address(), nb, (c * 7) % 64, NPAGES]
            crt[cx][cy] = [nb]
            wrt[cx][cy] = [out.buffer_address(), nb, c * nb]
            c += 1
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_zcollapse.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_zcollapse.cpp", cct, crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(KDIR / "writer_fslice.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(CB_V, TILE_B, NPLANE + 2 * shift), cb(CB_MASK, TILE_B, NPLANE),
                           cb(CB_OUT, TILE_B, 4 * ncopy)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"nplane": NPLANE, "shifts": list(SHIFTS), "arms": {}}
    try:
        rng = np.random.default_rng(71)
        vol = rng.integers(-100, 100, size=(NPAGES * 32, 32)).astype(np.float32)
        vt = torch.from_numpy(vol).to(torch.bfloat16)
        voln = vt.to(torch.float64).numpy().reshape(NPAGES, 32, 32)
        # THE VOLUME DOES NOT FIT L1 AT ANY REAL BOX. The padded half-volume is 268 MB at box 256,
        # 906 MB at 384 and 2.147 GB at 512, against 130 x 1.5 MB = 195 MB of chip L1. What IS
        # L1-resident is the sliding plane WINDOW -- 28 tiles per core is 7.5 MB chip-wide -- so the
        # slab loop of section 4.3 streams the volume through L1 and the reuse happens inside it.
        # Measuring both bounds: an L1 source is the steady state where the slab is already resident,
        # a DRAM source is the pessimistic case where every entering plane comes from DRAM. The truth
        # is between, and closer to the L1 end as the walk gets shallower.
        vols = {}
        for tag, bt in (("l1", ttnn.BufferType.L1), ("dram", ttnn.BufferType.DRAM)):
            vols[tag] = ttnn.from_torch(
                vt.reshape(1, 1, NPAGES * 32, 32), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                device=dev, memory_config=ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, bt))
        v = vols["l1"]
        a, b = 0.43, 0.29
        mk = np.zeros((NPLANE, 32, 32), dtype=np.float32)
        for X in range(32):
            for Y in range(32):
                z = a * X + b * Y
                z0 = int(np.floor(z))
                t = z - z0
                if 0 <= z0 < NPLANE:
                    mk[z0, X, Y] = 1 - t
                if 0 <= z0 + 1 < NPLANE:
                    mk[z0 + 1, X, Y] = t
        m = ttnn.from_torch(torch.from_numpy(mk.reshape(1, 1, NPLANE * 32, 32)).to(torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        mkn = torch.from_numpy(mk).to(torch.bfloat16).to(torch.float64).numpy()

        base = None
        for shift in SHIFTS:
            # Correctness: output tile t sums the masks against planes t*shift .. t*shift+NPLANE-1.
            NCHK = 3
            out1 = ttnn.from_torch(torch.zeros(1, 1, 32 * NCHK, 32).to(torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, v, m, out1, 1, 1, NCHK, shift)
            ttnn.generic_op([v, m, out1], pd)
            ttnn.synchronize_device(dev)
            g = ttnn.to_torch(out1).reshape(NCHK, 32, 32).to(torch.float64).numpy()
            rels = []
            for tt in range(NCHK):
                s0 = tt * shift
                ref = np.einsum("pxy,pxy->xy", mkn, voln[s0:s0 + NPLANE])
                rels.append(np.linalg.norm(g[tt] - ref) / max(np.linalg.norm(ref), 1e-300))
            rel = float(max(rels))
            ttnn.deallocate(out1)

            nx, ny = 13, 10
            n = nx * ny
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n, 32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, v, m, out, nx, ny, NB, shift)
            ttnn.generic_op([v, m, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([v, m, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            ns = best * 1e9 / NB
            if base is None:
                base = ns
            per_slice = ns * TILES_PER_DIRECTION / n / PSI_PER_DIRECTION
            tot = per_slice + STAGE2_NS_PER_SLICE_PER_CORE
            res["arms"][str(shift)] = {
                "max_rel_l2": rel, "ns_per_tile_per_core": ns, "speedup_vs_no_reuse": base / ns,
                "stage1_ns_per_slice_per_core": per_slice, "projection_ns_per_slice_per_core": tot,
                "projection_k_slices_per_s": 1e9 / tot / 1e3,
                "projection_pct_of_floor": 100 * (1e9 / tot) / FLOOR_SLICES_S}
            print(f"shift {shift:3d}: rel L2 {rel:.3e}  {ns:9.1f} ns/tile ({base/ns:5.2f}x)  "
                  f"stage1 {per_slice:7.1f} ns/slice -> projection {1e9/tot/1e3:7.1f} k slices/s "
                  f"({100*(1e9/tot)/FLOOR_SLICES_S:4.1f}% of floor)", flush=True)
            # The general shear needs stage 1 to emit 8 replicated copies. Cost probe: bytes right,
            # contents of copies 1..7 wrong, so only the timing is used.
            if shift == 7:
                o8 = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n * 8, 32).to(torch.bfloat16),
                                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                pd8 = build(dev, v, m, o8, nx, ny, NB, shift, 8)
                ttnn.generic_op([v, m, o8], pd8)
                ttnn.synchronize_device(dev)
                b8 = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([v, m, o8], pd8)
                    ttnn.synchronize_device(dev)
                    b8 = min(b8, time.perf_counter() - t0)
                ns8 = b8 * 1e9 / NB
                ps8 = ns8 * TILES_PER_DIRECTION / n / PSI_PER_DIRECTION
                res["replicated_stage1"] = {"ns_per_tile_per_core": ns8,
                                            "ns_per_slice_per_core": ps8, "factor_vs_1copy": ns8 / ns}
                print(f"   stage 1 emitting 8 replicated copies: {ns8:9.1f} ns/tile "
                      f"({ns8/ns:5.2f}x)  -> {ps8:7.1f} ns/slice/core", flush=True)
                ttnn.deallocate(o8)
            # Same arm from DRAM: the pessimistic bound on where the entering planes come from.
            if shift in (7, 28):
                od = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n, 32).to(torch.bfloat16),
                                     dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
                pdd = build(dev, vols["dram"], m, od, nx, ny, NB, shift)
                ttnn.generic_op([vols["dram"], m, od], pdd)
                ttnn.synchronize_device(dev)
                bd = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([vols["dram"], m, od], pdd)
                    ttnn.synchronize_device(dev)
                    bd = min(bd, time.perf_counter() - t0)
                nsd = bd * 1e9 / NB
                psd = nsd * TILES_PER_DIRECTION / n / PSI_PER_DIRECTION
                res["arms"][str(shift)]["ns_dram_source"] = nsd
                res["arms"][str(shift)]["dram_over_l1"] = nsd / ns
                print(f"      DRAM source: {nsd:9.1f} ns/tile ({nsd/ns:5.2f}x L1)  -> "
                      f"stage1 {psd:7.1f} ns/slice/core", flush=True)
                ttnn.deallocate(od)
            json.dump(res, open(HERE / "fslice_reuse.json", "w"), indent=1)
            ttnn.deallocate(out)
    finally:
        ttnn.close_device(dev)


main()
