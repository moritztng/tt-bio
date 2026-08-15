#!/usr/bin/env python3
"""E8 -- what one tile op costs in fp32, because E4c's free budget was measured in bfloat16.

E4c concluded that the trilinear blend hides under the exact-trilinear gather with 14.6x of headroom:
16 tile ops per assembly-pair were free at 1204-1218 ns, and the real kernel was priced at ~1.09.
That whole result was measured with `ttnn.bfloat16` CBs at HiFi4. The coarse projection kernel cannot
run in bfloat16 -- its parity gate is a 1e-5 relative residual against RELION's own kernel and bf16
carries 8 mantissa bits, i.e. ~4e-3. So the free budget has to be restated in float32, which is the
format the kernel will actually use, and the fp32/bf16 ratio is what decides whether the assembled
kernel stays gather-bound.

The arm is deliberately narrow: one tile in, `ops` back-to-back `mul_tiles` into DST, one pack, one
tile out, `outer` times. Same compute kernel E4c used, so the two numbers are directly comparable.
Sweeping `ops` and fitting the slope gives ns per tile op; the intercept is the per-iteration
acquire/commit/pack cycle, which the kernel pays anyway.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
E4KDIR = HERE / "kernels"
IN_CB, SC_CB, OUT_CB = 0, 2, 16
OUTER = 5000
OPS = (0, 8, 16, 32, 64, 128)
import os
KERN = os.environ.get("E8_KERNEL", "compute_e4_blend.cpp")

ARMS = (
    ("bf16_hifi4", ttnn.bfloat16, 2048, ttnn.MathFidelity.HiFi4, False),
    ("bf16_lofi", ttnn.bfloat16, 2048, ttnn.MathFidelity.LoFi, False),
    ("fp32_hifi4", ttnn.float32, 4096, ttnn.MathFidelity.HiFi4, True),
    ("fp32_lofi", ttnn.float32, 4096, ttnn.MathFidelity.LoFi, True),
)


def build(dev, x, out, fmt, tb, fid, fp32acc, ops):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def cb(i, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=fmt, page_size=tb)
        return ttnn.CBDescriptor(total_size=d * tb, core_ranges=cg, format_descriptors=[f])

    sa = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    da = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [x.buffer_address(), c]
            crt[cx][cy] = [0]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = fid
    cc.fp32_dest_acc_en = fp32acc
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_e8_fill.cpp", [IN_CB, tb] + sa, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / KERN, [IN_CB, OUT_CB, ops, OUTER, SC_CB], crt, cc),
        mk(KDIR / "writer_e8_drain.cpp", [OUT_CB, tb] + da, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB, 2), cb(SC_CB, 2), cb(OUT_CB, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"outer": OUTER, "ops_sweep": OPS, "arms": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["grid"] = [g.x, g.y]
        res["ncores"] = nc
        for name, fmt, tb, fid, fp32acc in ARMS:
            t = 0.1 * torch.randn(1, 1, 32 * nc, 32)
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
            x = ttnn.from_torch(t, dtype=fmt, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32), dtype=fmt,
                                  layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            arm = {}
            for ops in OPS:
                pd = build(dev, x, out, fmt, tb, fid, fp32acc, ops)
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, out], pd)
                    ttnn.synchronize_device(dev)
                    best = min(best, time.perf_counter() - t0)
                ns = best / OUTER * 1e9
                arm[ops] = round(ns, 2)
                print("%-12s ops=%-4d %8.2f ns/iter" % (name, ops, ns), flush=True)
            nz = bool(ttnn.to_torch(out).abs().sum() > 0)
            # slope over the linear tail, 32..128
            hi, lo = arm[128], arm[32]
            slope = (hi - lo) / (128 - 32)
            arm["ns_per_tile_op"] = round(slope, 3)
            arm["nonzero_out"] = nz
            res["arms"][name] = arm
            print("%-12s -> %.3f ns per tile op, out nonzero=%s" % (name, slope, nz), flush=True)
            ttnn.deallocate(x)
            ttnn.deallocate(out)
        b = res["arms"]["bf16_hifi4"]["ns_per_tile_op"]
        f = res["arms"]["fp32_hifi4"]["ns_per_tile_op"]
        res["fp32_over_bf16_hifi4"] = round(f / b, 3) if b else None
        print("fp32/bf16 at HiFi4: %s" % res["fp32_over_bf16_hifi4"], flush=True)
    finally:
        ttnn.close_device(dev)
    (HERE / ("e8_tileop_rate_" + KERN.replace(".cpp", "") + ".json")).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
