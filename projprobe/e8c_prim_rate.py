#!/usr/bin/env python3
"""E8c -- what each primitive in the coarse kernel's op budget actually costs, in fp32.

E8 priced one primitive (mul_tiles) at 61.10 ns and the design's §4.3 budget is written in units of
it. That budget also spends transpose_wh_tile, reduce_tile in both directions, the bcast_rows family
and matmul_tiles, none of which E8 measured. P2 pre-registers each at <= 3 mul_tiles; this is the arm
that grades it, and a primitive that comes in above 3 forces §4.3 to be re-costed before the kernel is
written rather than after.

Same loop, same sweep, same slope fit as e8_tileop_rate.py, so the numbers subtract directly against
its baseline. fp32 + HiFi4 + fp32 dest-acc only: that is the configuration the parity gate forces, and
E8 already showed fp32 costs 1.004x bf16 on the FPU, so the other three arms would say nothing new.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
IN_CB, SC_CB, OUT_CB = 0, 2, 16
OUTER = 5000
OPS = (0, 32, 64, 128)
FMT, TB = ttnn.float32, 4096

PRIMS = (
    (0, "mul_tiles"),
    (1, "transpose_wh_tile"),
    (2, "reduce_sum_col"),
    (3, "reduce_sum_row"),
    (4, "mul_tiles_bcast_rows"),
    (5, "add_tiles_bcast_rows"),
    (6, "mul_tiles_bcast_cols"),
    (7, "matmul_tiles"),
    (8, "trunc_tile_sfpu"),
    (9, "frac_tile_sfpu"),
)


def build(dev, x, out, ops, prim):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def cb(i, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=FMT, page_size=TB)
        return ttnn.CBDescriptor(total_size=d * TB, core_ranges=cg, format_descriptors=[f])

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
    cc.math_fidelity = ttnn.MathFidelity.HiFi4
    cc.fp32_dest_acc_en = True
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_e8_fill.cpp", [IN_CB, TB] + sa, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_e8_prim.cpp", [IN_CB, OUT_CB, ops, OUTER, SC_CB, prim], crt, cc),
        mk(KDIR / "writer_e8_drain.cpp", [OUT_CB, TB] + da, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB, 2), cb(SC_CB, 2), cb(OUT_CB, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"outer": OUTER, "ops_sweep": OPS, "arm": "fp32_hifi4_fp32acc", "prims": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["grid"], res["ncores"] = [g.x, g.y], nc
        t = 0.1 * torch.randn(1, 1, 32 * nc, 32)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(t, dtype=FMT, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32), dtype=FMT,
                              layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        for prim, name in PRIMS:
            row = {}
            for ops in OPS:
                pd = build(dev, x, out, ops, prim)
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, out], pd)
                    ttnn.synchronize_device(dev)
                    best = min(best, time.perf_counter() - t0)
                row[ops] = round(best / OUTER * 1e9, 2)
            row["ns_per_tile_op"] = round((row[128] - row[32]) / (128 - 32), 3)
            row["nonzero_out"] = bool(ttnn.to_torch(out).abs().sum() > 0)
            res["prims"][name] = row
            print("%-22s %8.3f ns/op   raw=%s nonzero=%s"
                  % (name, row["ns_per_tile_op"], [row[o] for o in OPS], row["nonzero_out"]),
                  flush=True)
        base = res["prims"]["mul_tiles"]["ns_per_tile_op"]
        res["mul_tiles_baseline_ns"] = base
        res["in_mul_tiles"] = {k: round(v["ns_per_tile_op"] / base, 2)
                               for k, v in res["prims"].items()}
        print("in mul_tiles units: %s" % res["in_mul_tiles"], flush=True)
    finally:
        ttnn.close_device(dev)
    (HERE / "e8c_prim_rate.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
