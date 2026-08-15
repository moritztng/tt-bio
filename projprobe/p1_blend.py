#!/usr/bin/env python3
"""Phase 1(iii) -- the trilinear blend on device, graded against numpy.

Seven lerps in RELION's own association order, a + (b - a) * f, turning eight gathered corners and
three fractions into the interpolated reference. Every op is an SFPU DST-to-DST binary under
unpack_to_dest, the only set §8.4 measured exact; this stage multiplies model values by weights, so
an FPU path's ~11-bit operand truncation would land straight on diff2.

Inputs are synthetic but shaped exactly as the assembled kernel will supply them: eight corner tiles
in the gather reader's slot order, and fx/fy/fz/mask/sgn from the coordinate stage, all in the
paired-column layout where each pixel owns two adjacent columns, re and im.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
CB_SLOT, CB_DEN, CB_C, CB_S, CB_OUT = 0, 1, 2, 3, 16
TB = 4096
N_S = 26          # exactly what one block pushes: 7 lerps x 3, plus 3 for the sign, plus 2
N_BLOCKS = 4


def lerp(a, b, f):
    return a + (b - a) * f


def main():
    rng = np.random.default_rng(7)
    sh = (N_BLOCKS, 32, 32)
    c = (rng.standard_normal((8,) + sh) * 0.5).astype(np.float32)
    fx = rng.random(sh).astype(np.float32)
    fy = rng.random(sh).astype(np.float32)
    fz = rng.random(sh).astype(np.float32)
    mask = (rng.random(sh) > 0.25).astype(np.float32)
    sgn = np.where(rng.random(sh) < 0.5, np.float32(-1), np.float32(1)).astype(np.float32)
    # Every per-pair quantity is constant across a pixel's two columns, which is what the coordinate
    # stage produces; the blend must not depend on that, but the oracle has to match it.
    for a in (fx, fy, fz, mask, sgn):
        a[:, :, 1::2] = a[:, :, 0::2]

    odd = np.zeros((32, 32), dtype=np.float32)
    odd[:, 1::2] = 1.0

    dx00 = lerp(c[0], c[1], fx)
    dx10 = lerp(c[2], c[3], fx)
    dx01 = lerp(c[4], c[5], fx)
    dx11 = lerp(c[6], c[7], fx)
    ref = lerp(lerp(dx00, dx10, fy), lerp(dx01, dx11, fy), fz)
    want = ref * (1.0 + (sgn - 1.0) * odd[None]) * mask

    slot_np = np.stack([c[s][b] for b in range(N_BLOCKS) for s in range(8)])
    den_np = np.stack([a[b] for b in range(N_BLOCKS) for a in (fx, fy, fz, mask, sgn)])
    c_np = np.stack([np.ones((32, 32), dtype=np.float32), odd])

    dev = ttnn.open_device(device_id=0)
    try:
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        to = lambda a: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)).reshape(1, 1, -1, 32),
                                       dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=dev, memory_config=mc)
        ts, td, tc = to(slot_np), to(den_np), to(c_np)
        tout = to(np.zeros((N_BLOCKS, 32, 32), dtype=np.float32))

        cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        def cb(i, depth):
            f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
            return ttnn.CBDescriptor(total_size=depth * TB, core_ranges=cg, format_descriptors=[f])

        sa = list(ttnn.TensorAccessorArgs(ts).get_compile_time_args())
        da = list(ttnn.TensorAccessorArgs(td).get_compile_time_args())
        ca = list(ttnn.TensorAccessorArgs(tc).get_compile_time_args())
        oa = list(ttnn.TensorAccessorArgs(tout).get_compile_time_args())
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rrt[0][0] = [ts.buffer_address(), td.buffer_address(), tc.buffer_address()]
        crt[0][0] = [0]
        wrt[0][0] = [tout.buffer_address()]
        mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
            kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
        cc = ttnn.ComputeConfigDescriptor()
        cc.math_fidelity = ttnn.MathFidelity.HiFi4
        cc.fp32_dest_acc_en = True
        cc.unpack_to_dest_mode = [ttnn.UnpackToDestMode.UnpackToDestFp32] * 64
        pd = ttnn.ProgramDescriptor(kernels=[
            mk(KDIR / "reader_p1_blend.cpp",
               [CB_SLOT, CB_DEN, CB_C, TB, N_BLOCKS] + sa + da + ca, rrt,
               ttnn.ReaderConfigDescriptor()),
            mk(KDIR / "compute_p1_blend.cpp", [N_BLOCKS], crt, cc),
            mk(KDIR / "writer_p1_out.cpp", [CB_OUT, TB, N_BLOCKS, 1] + oa, wrt,
               ttnn.WriterConfigDescriptor()),
        ], semaphores=[], cbs=[cb(CB_SLOT, 8), cb(CB_DEN, 5), cb(CB_C, 2), cb(CB_S, N_S),
                               cb(CB_OUT, 2)])
        ttnn.generic_op([ts, td, tc, tout], pd)
        ttnn.synchronize_device(dev)
        got = ttnn.to_torch(tout).numpy().reshape(N_BLOCKS, 32, 32)
    finally:
        ttnn.close_device(dev)

    nz_want, nz_got = float((want != 0).mean()), float((got != 0).mean())
    print("non-zero fraction: want %.3f  got %.3f" % (nz_want, nz_got), flush=True)
    assert nz_want > 0.5 and nz_got > 0.5, "vacuous comparison -- one side is mostly zero"
    e = float(np.abs(got - want).max())
    rel = e / float(np.abs(want).max())
    print("blend max_abs_err %.6e   max_rel %.6e   bit-exact %s"
          % (e, rel, e == 0.0), flush=True)
    (HERE / "p1_blend.json").write_text(json.dumps(
        {"max_abs_err": e, "max_rel": rel, "bit_exact": e == 0.0,
         "nonzero": {"want": nz_want, "got": nz_got}}, indent=1))


if __name__ == "__main__":
    main()
