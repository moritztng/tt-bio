#!/usr/bin/env python3
"""S2b -- the matmul roof on Blackhole, measured with a realistic operand pattern.

S1b reported 50.37 ns per 32x32x32 fp32 tile matmul, which is 170 TFLOP/s across 130 cores and
above this card's published bf16 matmul rate, so it cannot be a true fp32 figure. This screen finds
out what was being reused, by sweeping the one thing S1b held fixed (the operand tiles) alongside
the two knobs that actually set the matmul rate (math fidelity and input dtype).

The rate is read off the SLOPE in K, not off a single point, so fixed per-iteration costs -- the L1
round trip, the CB push/pop, the loop -- difference out and do not inflate the roof.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).resolve().parent / "s1b_kernels"
IN_CB, OUT_CB, SCRATCH_CB = 0, 16, 24
NT = 8
OUTER = 20000
KS = (4, 16)

FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2,
       "HiFi4": ttnn.MathFidelity.HiFi4}
DT = {"fp32": (ttnn.float32, torch.float32, 4), "bf16": (ttnn.bfloat16, torch.bfloat16, 2)}


def build(dev, x, out, K, fid, dt, walk):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    tdt, _, nb = DT[dt]
    tb = 32 * 32 * nb

    def cb(i, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=tdt, page_size=tb)
        return ttnn.CBDescriptor(total_size=d * tb, core_ranges=cg, format_descriptors=[f])

    rct = [IN_CB, tb, NT] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    wct = [OUT_CB, tb] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [x.buffer_address(), NT * c]
            crt[cx][cy] = [OUTER]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    mk = lambda s, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(KDIR / s), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk("reader_s1b.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk("writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
        ttnn.KernelDescriptor(
            kernel_source=str(KDIR / "compute_s2b.cpp"),
            source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
            core_ranges=cg,
            compile_time_args=[IN_CB, SCRATCH_CB, OUT_CB, K, NT, walk],
            runtime_args=crt,
            config=ttnn.ComputeConfigDescriptor(math_fidelity=fid, fp32_dest_acc_en=True)),
    ], semaphores=[], cbs=[cb(IN_CB, NT), cb(OUT_CB, 2), cb(SCRATCH_CB, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"outer": OUTER, "ks": list(KS), "nt": NT, "points": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        n = g.x * g.y
        res["ncores"] = n
        torch.manual_seed(0)
        tensors = {}
        for dt, (tdt, tor, _) in DT.items():
            xt = (0.1 * torch.randn(1, 1, 32 * NT * n, 32)).to(tor)
            tensors[dt] = (
                ttnn.from_torch(xt, dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(torch.zeros(1, 1, 32 * n, 32).to(tor), dtype=tdt,
                                layout=ttnn.TILE_LAYOUT, device=dev),
            )
        for dt in DT:
            x, out = tensors[dt]
            for fname, fid in FID.items():
                for walk in (0, 1):
                    ts = {}
                    for K in KS:
                        pd = build(dev, x, out, K, fid, dt, walk)
                        ttnn.generic_op([x, out], pd)
                        ttnn.synchronize_device(dev)
                        best = float("inf")
                        for _ in range(5):
                            t0 = time.perf_counter()
                            ttnn.generic_op([x, out], pd)
                            ttnn.synchronize_device(dev)
                            best = min(best, time.perf_counter() - t0)
                        ts[K] = best * 1e9 / OUTER
                    slope = (ts[KS[-1]] - ts[KS[0]]) / (KS[-1] - KS[0])
                    tfs = 130 * 2 * 32 ** 3 / slope / 1e3     # 2*32^3 flops per tile matmul
                    key = f"{dt}/{fname}/walk={walk}"
                    res["points"][key] = {"ns_per_matmul": slope,
                                          "per_iter_ns": {str(k): v for k, v in ts.items()},
                                          "chip_tflops": tfs}
                    print(f"{key:22s} {slope:7.2f} ns/tile-matmul   {tfs:7.1f} TFLOP/s chip",
                          flush=True)
                    json.dump(res, open(Path(__file__).resolve().parent / "screen_s2b.json", "w"),
                              indent=1)
    finally:
        ttnn.close_device(dev)


main()
