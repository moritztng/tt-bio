#!/usr/bin/env python3
"""Stage 1, the z-collapse: the last piece before a projection rate exists.

W[X, Y] = V[X, Y, a*X + b*Y] by linear interpolation along z. Every cell of a 32x32 (X, Y) tile draws
from the two z-planes bracketing its own z, and z varies across the tile, so the band of planes the
tile touches is wide even though only two matter per cell. S3 measured that band over real HEALPix
directions with the section-4.3 axis permutation: mean 28.27 planes, p95 51.05, max 62.16.

Written as sum_p mask_p * V_p with the masks host-computed and fixed for the direction, which is the
same move that took stage 2's SFPU bill down by 1.34x in section 18.

Swept over plane counts because the band is a distribution, not a constant: the mean, the p95 and the
max all occur in a real refinement and the per-slice cost is what the distribution averages to.

Stage 1's cost is amortised 96-fold against stage 2, because W depends only on the viewing direction
while stage 2 applies the in-plane rotation, and healpix order 4 has 96 psi per direction.
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
PLANES = (2, 8, 28, 51, 62)          # 2 = the cells own bracket; then S3's mean, p95 and max
PSI_PER_DIRECTION = 96
TILES_PER_SLICE = 25736 / 1024.0
FLOOR_SLICES_S = 3.20e6
# At box 256 the padded plane is 512 x 512, so a direction's W is 256 tiles per real component.
TILES_PER_DIRECTION = (512 // 32) ** 2 * 2


def build(dev, v, m, out, nx, ny, nplane, nb, nmul=None):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(nx - 1, ny - 1))])

    def cb(i, page, depth):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=page)
        return ttnn.CBDescriptor(total_size=depth * page, core_ranges=cg, format_descriptors=[f])

    rct = ([CB_V, CB_MASK, nplane, TILE_B, BARRIER_EVERY, nplane, nplane]
           + list(ttnn.TensorAccessorArgs(v).get_compile_time_args())
           + list(ttnn.TensorAccessorArgs(m).get_compile_time_args()))
    cct = [CB_V, CB_MASK, CB_OUT, nplane, nplane if nmul is None else nmul]
    wct = [CB_OUT, TILE_B, 1] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
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
    ], semaphores=[], cbs=[cb(CB_V, TILE_B, 2 * nplane), cb(CB_MASK, TILE_B, nplane),
                           cb(CB_OUT, TILE_B, 4)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"planes": list(PLANES), "psi_per_direction": PSI_PER_DIRECTION,
           "tiles_per_direction": TILES_PER_DIRECTION, "arms": {}}
    try:
        rng = np.random.default_rng(61)
        vol = rng.integers(-100, 100, size=(NPAGES * 32, 32)).astype(np.float32)
        vt = torch.from_numpy(vol).to(torch.bfloat16)
        voln = vt.to(torch.float64).numpy().reshape(NPAGES, 32, 32)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        v = ttnn.from_torch(vt.reshape(1, 1, NPAGES * 32, 32), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

        for nplane in PLANES:
            # Interpolation masks: cell (X, Y) has z = a*X + b*Y, so planes floor(z) and floor(z)+1
            # carry 1-t and t and every other plane carries zero.
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

            out1 = ttnn.from_torch(torch.zeros(1, 1, 32, 32).to(torch.bfloat16), dtype=ttnn.bfloat16,
                                   layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, v, m, out1, 1, 1, nplane, 1)
            ttnn.generic_op([v, m, out1], pd)
            ttnn.synchronize_device(dev)
            g = ttnn.to_torch(out1).reshape(32, 32).to(torch.float64).numpy()
            ref = np.einsum("pxy,pxy->xy", mkn, voln[0:nplane])
            rel = float(np.linalg.norm(g - ref) / max(np.linalg.norm(ref), 1e-300))
            ttnn.deallocate(out1)

            nx, ny = 13, 10
            n = nx * ny
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n, 32).to(torch.bfloat16),
                                  dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd = build(dev, v, m, out, nx, ny, nplane, NB)
            ttnn.generic_op([v, m, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([v, m, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            ns = best * 1e9 / NB
            # Per slice per core: tiles_per_direction/ncores tiles, amortised over 96 psi.
            per_slice_ns = ns * TILES_PER_DIRECTION / n / PSI_PER_DIRECTION
            res["arms"][str(nplane)] = {"rel_l2": rel, "ns_per_tile_per_core": ns,
                                        "ns_per_slice_per_core_amortised": per_slice_ns}
            print(f"nplane {nplane:3d}: rel L2 {rel:.3e}   {ns:9.1f} ns/tile/core   "
                  f"-> {per_slice_ns:7.1f} ns per slice per core after 96x amortisation", flush=True)
            # Cost probe: same reads, one multiply. Wrong result on purpose; the difference from the
            # full arm is the arithmetic, and what is left is the reads.
            out2 = ttnn.from_torch(torch.zeros(1, 1, 32 * NB * n, 32).to(torch.bfloat16),
                                   dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
            pd2 = build(dev, v, m, out2, nx, ny, nplane, NB, 1)
            ttnn.generic_op([v, m, out2], pd2)
            ttnn.synchronize_device(dev)
            b2 = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([v, m, out2], pd2)
                ttnn.synchronize_device(dev)
                b2 = min(b2, time.perf_counter() - t0)
            ns_reads = b2 * 1e9 / NB
            res["arms"][str(nplane)]["ns_reads_only"] = ns_reads
            res["arms"][str(nplane)]["read_share_pct"] = 100.0 * ns_reads / ns
            print(f"            reads-only {ns_reads:9.1f} ns  -> reads are "
                  f"{100.0*ns_reads/ns:4.1f}% of the z-collapse", flush=True)
            ttnn.deallocate(out2)
            json.dump(res, open(HERE / "fslice_stage1.json", "w"), indent=1)
            ttnn.deallocate(out)
            ttnn.deallocate(m)

        # Stage 2 fused measured 4045.5 ns per FINAL output tile per core; a slice is 25.1 such tiles
        # over 130 cores, so 0.193 tiles per core per slice.
        s2 = 4045.5 * TILES_PER_SLICE / 130
        mean_arm = res["arms"].get("28")
        if mean_arm:
            s1 = mean_arm["ns_per_slice_per_core_amortised"]
            tot = s1 + s2
            print(f"\nstage 2 fused: {s2:7.1f} ns per slice per core")
            print(f"stage 1 at the S3 mean band of 28: {s1:7.1f} ns  ({100*s1/tot:4.1f}% of the total)")
            print(f"projection total: {tot:7.1f} ns per slice per core -> "
                  f"{1e9/tot/1e3:7.1f} k slices/s ({100*(1e9/tot)/FLOOR_SLICES_S:4.1f}% of floor)")
            res["stage2_ns_per_slice_per_core"] = s2
            res["stage1_ns_per_slice_per_core"] = s1
            res["projection_k_slices_per_s"] = 1e9 / tot / 1e3
            res["projection_pct_of_floor"] = 100 * (1e9 / tot) / FLOOR_SLICES_S
            json.dump(res, open(HERE / "fslice_stage1.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)


main()
