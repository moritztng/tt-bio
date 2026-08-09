#!/usr/bin/env python3
"""W4 probe 2b: the same ops, timed amortized, plus a generic_op smoke test on Blackhole.

Probe 2a timed one op per synchronize/synchronize region and the numbers did not add up:
`transpose(-2,-1)` 0.0696 + `permute(0,2,1,3)` 0.1759 = 0.2455 ms, but the two of them
back to back in one region measured 0.1781 ms. A per-region cost that large swamps a
26 MB L1 op, so every absolute number from 2a is inflated and only the ratios survive.
Here each op is issued `--reps` times inside ONE region, so what is reported is the
marginal per-op cost.

Second question, decisive for whether this leg can build anything at all:
`ttnn.generic_op` is present in the production 0.67.4 pin, which would mean custom
kernels with no tt-metal rebuild -- but upstream's own generic_op test is marked
`skip_for_blackhole("Not tested / built for Blackhole")`. So run one on this card and see.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import ttnn

from tt_bio.tenstorrent import get_device

L1 = ttnn.L1_MEMORY_CONFIG


def amort(dev, fn, reps=16, warm=2, iters=5):
    """Marginal per-op ms: `reps` issues inside one sync..sync region."""
    for _ in range(warm):
        for _ in range(reps):
            r = fn()
            if isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(reps):
            r = fn()
            if isinstance(r, ttnn.Tensor):
                ttnn.deallocate(r)
        ttnn.synchronize_device(dev)
        ts.append((time.perf_counter() - t0) * 1e3 / reps)
    return sorted(ts)[len(ts) // 2]


def generic_op_smoke(dev, num_tiles=64):
    """One eltwise-exp generic_op, upstream's own recipe, on this Blackhole card."""
    shape = [1, num_tiles, 32, 32]
    data = torch.rand(shape).to(torch.bfloat16)
    mc = ttnn.DRAM_MEMORY_CONFIG
    inp = ttnn.from_torch(data, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                          memory_config=mc)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape(shape), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, dev, mc)
    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(7, 7))])
    (_, core_grid, cg1, cg2, per_core, _) = ttnn.split_work_to_cores(all_cores, num_tiles)
    assert len(cg2.ranges()) == 0
    page = 2 * 1024
    fmt = lambda idx: ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.bfloat16,
                                              page_size=page)
    in_cb = ttnn.CBDescriptor(total_size=2 * page, core_ranges=core_grid,
                              format_descriptors=[fmt(0)])
    out_cb = ttnn.CBDescriptor(total_size=2 * page, core_ranges=core_grid,
                               format_descriptors=[fmt(16)])
    r_ct = ttnn.TensorAccessorArgs(inp).get_compile_time_args()
    w_ct = [16] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    r_rt, w_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    t = 0
    for cr in cg1.ranges():
        for x in range(cr.start.x, cr.end.x + 1):
            for y in range(cr.start.y, cr.end.y + 1):
                r_rt[x][y] = [inp.buffer_address(), per_core, t]
                w_rt[x][y] = [out.buffer_address(), per_core, t]
                t += per_core
    K = ttnn.KernelDescriptor
    reader = K(kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                             "reader_unary_interleaved_start_id.cpp",
               source_type=K.SourceType.FILE_PATH, core_ranges=core_grid,
               compile_time_args=r_ct, runtime_args=r_rt, config=ttnn.ReaderConfigDescriptor())
    writer = K(kernel_source="ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                             "writer_unary_interleaved_start_id.cpp",
               source_type=K.SourceType.FILE_PATH, core_ranges=core_grid,
               compile_time_args=w_ct, runtime_args=w_rt, config=ttnn.WriterConfigDescriptor())
    compute = K(kernel_source="tt_metal/kernels/compute/eltwise_sfpu.cpp",
                core_ranges=core_grid, compile_time_args=[per_core, 1],
                defines=[("SFPU_OP_EXP_INCLUDE", "1"),
                         ("SFPU_OP_CHAIN_0", "exp_tile_init(); exp_tile(0);")],
                runtime_args=[], config=ttnn.ComputeConfigDescriptor())
    pd = ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[],
                                cbs=[in_cb, out_cb])
    got = ttnn.to_torch(ttnn.generic_op([inp, out], pd)).float()
    ref = ttnn.to_torch(ttnn.exp(inp)).float()
    return dict(max_abs_diff=float((got - ref).abs().max()),
                exact=bool(torch.equal(got, ref)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=320)
    ap.add_argument("--c", type=int, default=64)
    ap.add_argument("--reps", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    dev = get_device()
    N, C = args.n, args.c
    torch.manual_seed(0)
    rows = []

    tin = torch.randn(1, N, N, C)
    x = ttnn.from_torch(tin, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                        memory_config=L1)
    xt = ttnn.transpose(x, -2, -1, memory_config=L1)
    gp = ttnn.from_torch(torch.randn(1, N, N, 4 * C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    ga = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    pa = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)
    mask = ttnn.from_torch(torch.ones(1, N, N, 1), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16, memory_config=L1)
    rw = 2 * (N * N * C * 2) / 1e6

    cases = [
        ("clone (L1 copy roof)", lambda: ttnn.clone(x, memory_config=L1), rw),
        ("permute(0,3,1,2) 'a'", lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=L1), rw),
        ("permute(0,3,2,1) 'b'", lambda: ttnn.permute(x, (0, 3, 2, 1), memory_config=L1), rw),
        ("transpose(-2,-1)  [whole-tile half]",
         lambda: ttnn.transpose(x, -2, -1, memory_config=L1), rw),
        ("permute(0,2,1,3)  [sub-tile half]",
         lambda: ttnn.permute(xt, (0, 2, 1, 3), memory_config=L1), rw),
        ("transpose(1,3) 'b' one op", lambda: ttnn.transpose(x, 1, 3, memory_config=L1), rw),
        ("chunk(4) of [1,N,N,4C]", lambda: ttnn.chunk(gp, chunks=4, dim=-1),
         2 * (N * N * 4 * C * 2) / 1e6),
        ("gate multiply(p,g,SIGMOID)",
         lambda: ttnn.multiply(pa, ga, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID],
                               memory_config=L1), 3 * (N * N * C * 2) / 1e6),
        ("mask multiply(a,mask_u)", lambda: ttnn.multiply(pa, mask, memory_config=L1), rw),
    ]
    print(f"\n=== amortized, {args.reps} issues per region, N={N} C={C} L1 ===", flush=True)
    for name, fn, mb in cases:
        try:
            ms = amort(dev, fn, reps=args.reps)
            gbs = mb / ms * 1000 / 1e3 if mb else None
            rows.append(dict(part=name, ms=round(ms, 4), moved_mb=round(mb, 2) if mb else None,
                             eff_gbs=round(gbs, 1) if gbs else None))
            print("  %-40s %8.4f ms  %7.1f MB %7.1f GB/s" % (name, ms, mb, gbs), flush=True)
        except Exception as e:
            print("  %-40s FAILED %s: %s" % (name, type(e).__name__,
                                             str(e).splitlines()[0][:100]), flush=True)

    print("\n=== generic_op smoke test on Blackhole ===", flush=True)
    try:
        res = generic_op_smoke(dev)
        print("  generic_op eltwise-exp:", res, flush=True)
        rows.append(dict(part="generic_op_smoke", **res))
    except Exception as e:
        print("  generic_op FAILED %s: %s" % (type(e).__name__, str(e)[:400]), flush=True)
        rows.append(dict(part="generic_op_smoke", error=f"{type(e).__name__}: {str(e)[:300]}"))

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dict(n=N, c=C, reps=args.reps, rows=rows),
                                            indent=2) + "\n")
    from tt_bio.tenstorrent import cleanup
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
