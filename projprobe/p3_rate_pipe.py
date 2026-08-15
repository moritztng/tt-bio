#!/usr/bin/env python3
"""Phase 3b -- how fast the ASSEMBLED kernel actually runs, against the composed estimate.

Every rate this leg has quoted for the fused pipeline is a composition of separately-measured parts:
E4b's gather rate times a read count, plus an SFPU op budget. perf-method-floor-screen-predict-then-
build exists because that kind of composition is where predictions go wrong, so this arm times the
real program -- coordinate stage, gather and blend, running together, on the same code §8.13 graded
bit-exact -- and reports ns per (orientation, pixel) pair.

Three caveats, none of them small, all of them in the direction of this being a LOWER bound on
throughput rather than an upper one:
  - ONE core and ONE dataflow RISC issues the gather. §4.2's design uses both, which E4b measured at
    1.97x, so the production reader should be roughly twice as fast per core.
  - the model is in the reading core's OWN L1, not a neighbour's shard, so this omits whatever the
    NoC hop costs in the 130-core layout.
  - the compare stage is not in this program (§8.12 grades it separately).

The point of this arm is the dataflow direction, not the arithmetic: §4.4 requires the addresses to
come from the compute unit rather than the reader, so the program runs compute -> reader -> compute
within a block. The compute kernel pushes an address tile, the reader gathers eight corners from its
L1-resident model, and the compute kernel blends them. Each stage was already graded on its own
(§8.8, §8.9, §8.10); what is new here is that they run as one program without deadlocking and still
land on RELION's answer.

The geometry is reduced -- a 16x16x16 model, 32 kB, so it fits one core's L1 -- because a single-core
arm cannot hold RELION's real 31.68 MB model, and §4.1's 130-core sharding is what makes that work in
production. The arithmetic is identical; only the sizes change.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
CB_E, CB_XY, CB_C, CB_S1, CB_ADDR, CB_SLOT, CB_S2, CB_MDL, CB_SCR, CB_OUT = 0, 1, 2, 3, 4, 5, 6, 7, 8, 16
TB = 4096
N_S1, N_S2 = 70, 31   # two blocks of dense scratch: the pipeline keeps b and b+1 live
MDLX = MDLY = MDLZ = 16
MDLXY = MDLX * MDLY
NVOX = MDLXY * MDLZ
INIT = -8
ORIGIN = -INIT * MDLXY - INIT * MDLX
MAXR2 = 49
N_BLOCKS = 200        # enough blocks that the per-block cost dominates program setup
PPT = 16


def main():
    rng = np.random.default_rng(3)
    mdl = (rng.standard_normal((NVOX, 2)) * 0.5).astype(np.float32)

    # Random proper rotations, and a small lattice, so every gathered cell stays inside the model.
    eul = np.zeros((N_BLOCKS, 32, 9), dtype=np.float32)
    for b in range(N_BLOCKS):
        for o in range(32):
            q = rng.standard_normal(4)
            q /= np.linalg.norm(q)
            w, i, j, k = q
            eul[b, o] = [1 - 2 * (j * j + k * k), 2 * (i * j - k * w), 2 * (i * k + j * w),
                         2 * (i * j + k * w), 1 - 2 * (i * i + k * k), 2 * (j * k - i * w),
                         2 * (i * k - j * w), 2 * (j * k + i * w), 1 - 2 * (i * i + j * j)]
    xs = rng.integers(-3, 4, size=N_BLOCKS * PPT).astype(np.float32)
    ys = rng.integers(-3, 4, size=N_BLOCKS * PPT).astype(np.float32)
    x = np.repeat(xs, 2)
    y = np.repeat(ys, 2)

    e_np = np.zeros((N_BLOCKS * 6, 32, 32), dtype=np.float32)
    xy_np = np.zeros((N_BLOCKS * 2, 32, 32), dtype=np.float32)
    for b in range(N_BLOCKS):
        for kk, cc in enumerate((0, 1, 3, 4, 6, 7)):
            e_np[b * 6 + kk] = eul[b][:, cc][:, None]
        xy_np[b * 2 + 0] = x[b * 32:(b + 1) * 32][None, :]
        xy_np[b * 2 + 1] = y[b * 32:(b + 1) * 32][None, :]

    odd = np.zeros((32, 32), dtype=np.float32)
    odd[:, 1::2] = 1.0
    c_np = np.zeros((9, 32, 32), dtype=np.float32)
    for i, v in enumerate((MDLXY, MDLX, ORIGIN, 8388607.0, 16777215.0, 1.0, -2.0, 1.0)):
        c_np[i] = np.float32(v)
    c_np[8] = odd

    # The oracle: RELION's interpolant, in the kernel's own order.
    want = np.zeros((N_BLOCKS, 32, 32), dtype=np.float32)
    for b in range(N_BLOCKS):
        e = eul[b]
        xb = x[b * 32:(b + 1) * 32][None, :]
        yb = y[b * 32:(b + 1) * 32][None, :]
        xp = e[:, 0:1] * xb + e[:, 1:2] * yb
        yp = e[:, 3:4] * xb + e[:, 4:5] * yb
        zp = e[:, 6:7] * xb + e[:, 7:8] * yb
        mask = ((xp * xp + yp * yp + zp * zp) < np.float32(MAXR2 + 1)).astype(np.float32)
        sgn = np.where(xp < 0, np.float32(-1), np.float32(1)).astype(np.float32)
        xf, yf, zf = xp * sgn, yp * sgn, zp * sgn
        x0, y0, z0 = np.floor(xf), np.floor(yf), np.floor(zf)
        fx, fy, fz = xf - x0, yf - y0, zf - z0
        base = (z0 * MDLXY + y0 * MDLX + x0 + ORIGIN).astype(np.int64)
        offs = (0, 1, MDLX, MDLX + 1, MDLXY, MDLXY + 1, MDLXY + MDLX, MDLXY + MDLX + 1)
        comp = np.zeros((32, 32), dtype=np.int64)
        comp[:, 0::2] = 0
        comp[:, 1::2] = 1
        cc = [mdl[np.clip(base + o, 0, NVOX - 1), comp] for o in offs]
        lp = lambda a, bb, f: a + (bb - a) * f
        r = lp(lp(lp(cc[0], cc[1], fx), lp(cc[2], cc[3], fx), fy),
               lp(lp(cc[4], cc[5], fx), lp(cc[6], cc[7], fx), fy), fz)
        want[b] = r * (1.0 + (sgn - 1.0) * odd) * mask

    dev = ttnn.open_device(device_id=0)
    try:
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        tile = lambda a: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)).reshape(1, 1, -1, 32),
                                         dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                         device=dev, memory_config=mc)
        te, txy, tc = tile(e_np), tile(xy_np), tile(c_np)
        tm = ttnn.from_torch(torch.from_numpy(mdl.reshape(-1)).reshape(1, 1, NVOX * 2 // 1024, 1024),
                             dtype=ttnn.float32, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev,
                             memory_config=mc)
        tout = tile(np.zeros((N_BLOCKS, 32, 32), dtype=np.float32))
        mdl_pages = NVOX * 2 // 1024

        cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        def cb(i, depth):
            f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
            return ttnn.CBDescriptor(total_size=depth * TB, core_ranges=cg, format_descriptors=[f])

        ea = list(ttnn.TensorAccessorArgs(te).get_compile_time_args())
        xa = list(ttnn.TensorAccessorArgs(txy).get_compile_time_args())
        ca = list(ttnn.TensorAccessorArgs(tc).get_compile_time_args())
        ma = list(ttnn.TensorAccessorArgs(tm).get_compile_time_args())
        oa = list(ttnn.TensorAccessorArgs(tout).get_compile_time_args())
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rrt[0][0] = [te.buffer_address(), txy.buffer_address(), tc.buffer_address(),
                     tm.buffer_address()]
        crt[0][0] = [int(np.float32(MAXR2 + 1).view(np.uint32))]
        wrt[0][0] = [tout.buffer_address()]
        mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
        cc_ = ttnn.ComputeConfigDescriptor()
        cc_.math_fidelity = ttnn.MathFidelity.HiFi4
        cc_.fp32_dest_acc_en = True
        cc_.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        pd = ttnn.ProgramDescriptor(kernels=[
            mk(KDIR / "reader_p3_fused.cpp",
               [CB_E, CB_XY, CB_C, CB_ADDR, CB_SLOT, CB_MDL, CB_SCR, TB, N_BLOCKS,
                MDLX, MDLXY, mdl_pages] + ea + xa + ca + ma, rrt, ttnn.ReaderConfigDescriptor()),
            mk(KDIR / "compute_p3_pipe.cpp", [N_BLOCKS], crt, cc_),
            mk(KDIR / "writer_p1_out.cpp", [CB_OUT, TB, N_BLOCKS, 1] + oa, wrt,
               ttnn.WriterConfigDescriptor()),
        ], semaphores=[], cbs=[cb(CB_E, 12), cb(CB_XY, 4), cb(CB_C, 9), cb(CB_S1, N_S1),
                               cb(CB_ADDR, 4), cb(CB_SLOT, 16), cb(CB_S2, N_S2),
                               cb(CB_MDL, mdl_pages), cb(CB_SCR, 1), cb(CB_OUT, 2)])
        import time
        ttnn.generic_op([te, txy, tc, tm, tout], pd)      # warm: compile and first run
        ttnn.synchronize_device(dev)
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            ttnn.generic_op([te, txy, tc, tm, tout], pd)
            ttnn.synchronize_device(dev)
            best = min(best, time.perf_counter() - t0)
        got = ttnn.to_torch(tout).numpy().reshape(N_BLOCKS, 32, 32)
        pairs = N_BLOCKS * 512
        ns_pair = best / pairs * 1e9
        print("fused wall %.3f ms for %d pairs -> %.1f ns/pair (1 core, 1 gather RISC, pipelined)"
              % (best * 1e3, pairs, ns_pair), flush=True)
        print("composed estimate for this configuration: 8 reads x 36.7 ns = 293.6 ns/pair gather, "
              "single-RISC", flush=True)
    finally:
        ttnn.close_device(dev)

    nz_w, nz_g = float((want != 0).mean()), float((got != 0).mean())
    print("non-zero fraction: want %.3f  got %.3f" % (nz_w, nz_g), flush=True)
    assert nz_w > 0.4 and nz_g > 0.4, "vacuous comparison -- one side is mostly zero"
    e = float(np.abs(got - want).max())
    r = e / float(np.abs(want).max())
    print("fused ref  max_abs %.6e  max_rel %.6e  bit-exact %s" % (e, r, e == 0.0), flush=True)
    (HERE / "p3_fused.json").write_text(json.dumps(
        {"max_abs": e, "max_rel": r, "bit_exact": e == 0.0,
         "nonzero": {"want": nz_w, "got": nz_g}, "n_blocks": N_BLOCKS}, indent=1))


if __name__ == "__main__":
    main()
