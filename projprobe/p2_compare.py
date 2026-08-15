#!/usr/bin/env python3
"""Phase 2 -- the squared-difference compare and the pixel accumulation, graded against numpy.

RELION's diff2 body for one orientation block and one translation:
    d2 = sum_pixels ( (ref_r - sh_r)^2 + (ref_i - sh_i)^2 ) * w

In the paired-column layout the real and imaginary differences are the even and odd columns of one
tile, so their squares are a single tile op and summing over both columns IS the |.|^2 -- the
component sum falls out of the layout rather than costing an op. Four SFPU ops per pixel block, all
on the exact set (§8.4).

What this arm grades is everything up to the last step: the accumulator over pixel blocks, which is
element-wise and therefore exact. The remaining fold of 32 columns into one number per orientation
cannot be done by an element-wise op; it is 6 orientation blocks x 9 translations per call and is
handled outside this kernel, so the grader does it in numpy and checks both the accumulator and the
folded answer.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
CB_REF, CB_SH, CB_W, CB_S, CB_OUT = 0, 1, 2, 3, 16
TB = 4096
N_BLOCKS = 8
N_S = 3 + 4 * (N_BLOCKS - 1)      # exactly what the kernel pushes; a larger CB straddles a wrap


def main():
    rng = np.random.default_rng(11)
    sh_ = (N_BLOCKS, 32, 32)
    ref = (rng.standard_normal(sh_) * 0.7).astype(np.float32)
    shift = (rng.standard_normal(sh_) * 0.7).astype(np.float32)
    w = rng.random(sh_).astype(np.float32)
    w[:, :, 1::2] = w[:, :, 0::2]          # the weight is per pixel, so equal on the pair's columns

    d = ref - shift
    acc_want = (d * d * w).sum(axis=0)      # element-wise across pixel blocks
    folded_want = acc_want.sum(axis=1)      # then the 32 columns -> one number per orientation

    dev = ttnn.open_device(device_id=0)
    try:
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        to = lambda a: ttnn.from_torch(torch.from_numpy(np.ascontiguousarray(a)).reshape(1, 1, -1, 32),
                                       dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                       device=dev, memory_config=mc)
        tr, ts, tw = to(ref), to(shift), to(w)
        tout = to(np.zeros((1, 32, 32), dtype=np.float32))

        cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

        def cb(i, depth):
            f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
            return ttnn.CBDescriptor(total_size=depth * TB, core_ranges=cg, format_descriptors=[f])

        ra = list(ttnn.TensorAccessorArgs(tr).get_compile_time_args())
        sa = list(ttnn.TensorAccessorArgs(ts).get_compile_time_args())
        wa = list(ttnn.TensorAccessorArgs(tw).get_compile_time_args())
        oa = list(ttnn.TensorAccessorArgs(tout).get_compile_time_args())
        rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        rrt[0][0] = [tr.buffer_address(), ts.buffer_address(), tw.buffer_address()]
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
            mk(KDIR / "reader_p2_compare.cpp",
               [CB_REF, CB_SH, CB_W, TB, N_BLOCKS] + ra + sa + wa, rrt,
               ttnn.ReaderConfigDescriptor()),
            mk(KDIR / "compute_p2_compare.cpp", [N_BLOCKS], crt, cc),
            mk(KDIR / "writer_p1_out.cpp", [CB_OUT, TB, 1, 1] + oa, wrt,
               ttnn.WriterConfigDescriptor()),
        ], semaphores=[], cbs=[cb(CB_REF, 2), cb(CB_SH, 2), cb(CB_W, 2), cb(CB_S, N_S),
                               cb(CB_OUT, 2)])
        ttnn.generic_op([tr, ts, tw, tout], pd)
        ttnn.synchronize_device(dev)
        acc_got = ttnn.to_torch(tout).numpy().reshape(32, 32)
    finally:
        ttnn.close_device(dev)

    nz_w, nz_g = float((acc_want != 0).mean()), float((acc_got != 0).mean())
    print("non-zero fraction: want %.3f  got %.3f" % (nz_w, nz_g), flush=True)
    assert nz_w > 0.5 and nz_g > 0.5, "vacuous comparison -- one side is mostly zero"

    e_acc = float(np.abs(acc_got - acc_want).max())
    r_acc = e_acc / float(np.abs(acc_want).max())
    folded_got = acc_got.sum(axis=1)
    e_f = float(np.abs(folded_got - folded_want).max())
    r_f = e_f / float(np.abs(folded_want).max())
    print("accumulator  max_abs %.6e  max_rel %.6e  bit-exact %s" % (e_acc, r_acc, e_acc == 0.0),
          flush=True)
    print("folded diff2 max_abs %.6e  max_rel %.6e  bit-exact %s" % (e_f, r_f, e_f == 0.0),
          flush=True)
    (HERE / "p2_compare.json").write_text(json.dumps(
        {"accumulator": {"max_abs": e_acc, "max_rel": r_acc, "bit_exact": e_acc == 0.0},
         "folded": {"max_abs": e_f, "max_rel": r_f, "bit_exact": e_f == 0.0},
         "n_blocks": N_BLOCKS, "nonzero": {"want": nz_w, "got": nz_g}}, indent=1))


if __name__ == "__main__":
    main()
