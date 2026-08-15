#!/usr/bin/env python3
"""E8d -- does an eltwise binary op accumulate into DST, or overwrite it?

The dense coordinate stage of the coarse kernel is a chain of small sums: xp = e0*x + e1*y, the sum
of three squares for the radius test, the address as z0*19900 + y0*100 + x0. If eltwise binary
accumulates into DST, each of those is one acquire and N ops. If it overwrites, each intermediate
needs its own pack-and-re-read, and the kernel is a longer chain of shorter cycles.

compute_e4_blend.cpp's comment claims accumulation, but that kernel is a timing screen whose own text
says the answer does not depend on the values. This one runs two identical ops into one DST slot on
known inputs and reads the tile back. matmul_tiles is the control: its header documents DST += C, so
it must come out at 2x.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
IN_CB, OUT_CB, TB = 0, 16, 4096


def run(dev, x, out, op):
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])

    def cb(i):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.float32, page_size=TB)
        return ttnn.CBDescriptor(total_size=2 * TB, core_ranges=cg, format_descriptors=[f])

    sa = list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    da = list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    rrt[0][0] = [x.buffer_address(), 0]
    crt[0][0] = [0]
    wrt[0][0] = [out.buffer_address(), 0]
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    cc = ttnn.ComputeConfigDescriptor()
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = True
    pd = ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_e8_fill.cpp", [IN_CB, TB] + sa, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_e8_dst.cpp", [IN_CB, OUT_CB, op], crt, cc),
        mk(KDIR / "writer_e8_drain.cpp", [OUT_CB, TB] + da, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB), cb(OUT_CB)])
    ttnn.generic_op([x, out], pd)
    ttnn.synchronize_device(dev)
    return ttnn.to_torch(out)[0, 0]


def main():
    dev = ttnn.open_device(device_id=0)
    res = {}
    try:
        t = torch.arange(1024, dtype=torch.float32).reshape(1, 1, 32, 32) * 0.01 + 0.5
        ref = t[0, 0]
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        for op, name, once in ((0, "mul_tiles", ref * ref),
                               (1, "add_tiles", ref + ref),
                               (2, "matmul_tiles", ref @ ref)):
            x = ttnn.from_torch(t, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT,
                                device=dev, memory_config=mc)
            out = ttnn.from_torch(torch.zeros(1, 1, 32, 32), dtype=ttnn.float32,
                                  layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            got = run(dev, x, out, op)
            e1 = (got - once).abs().max().item()
            e2 = (got - 2 * once).abs().max().item()
            rel = max(once.abs().max().item(), 1e-30)
            verdict = ("accumulates" if e2 < e1 else "overwrites")
            res[name] = {"err_vs_once": e1, "err_vs_twice": e2, "verdict": verdict,
                         "rel_err": min(e1, e2) / rel}
            print("%-14s vs 1x %.3e   vs 2x %.3e   -> %s (rel %.2e)"
                  % (name, e1, e2, verdict, res[name]["rel_err"]), flush=True)
            ttnn.deallocate(x)
            ttnn.deallocate(out)
    finally:
        ttnn.close_device(dev)
    (HERE / "e8d_dst_accum.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
