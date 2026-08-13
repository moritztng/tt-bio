#!/usr/bin/env python3
"""Screen S1b -- the kill gate for the fused-FFT thesis, plus S2 (matmul rate) in the same harness.

The question, restated. The L1-round model says a fused FFT stage costs one L1 read round and one
L1 write round, and that the arithmetic between them is free. Screen S1 measured 0.351 TFLOP/s for
ttnn-op-granularity eltwise, but that bounds a composite implementation only: every ttnn op pays a
full unpack + pack per tile, so the rate it reports is the rate of L1 round trips, not of
arithmetic. This screen puts K arithmetic operations inside ONE round trip and sweeps K.

    flat in K   -> arithmetic is free inside a round; the round-count model holds; build path C
    linear in K -> every operation costs a round; 0.351 TFLOP/s is the real ceiling for any
                   implementation; NO-GO on performance with the mechanism named

Five arms, each isolating one engine, all on tiles already resident in L1:
    sfpu_dest  K x mul_binary_tile(0,1,0)          DST-to-DST, the fused path's arithmetic
    fpu_mul    K x mul_tiles                        re-unpacks both operands per op
    fpu_matmul K x matmul_tiles 32x32x32            screen S2, the within-tile radix-32 stage
    transpose  K x transpose_wh_dest                the in-place DST transpose
    copy_only  no arithmetic                        the round-trip floor

Timing is host wall clock with device sync on both sides, over `outer` iterations per core on all
130 cores. `outer` is large enough that program launch is under 1% -- checked by the outer sweep.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
import ttnn

KDIR = Path(__file__).resolve().parent / "s1b_kernels"
IN_CB, OUT_CB, SCRATCH_CB = 0, 16, 24
TILE = 32
FP32_TILE_BYTES = TILE * TILE * 4

ARMS = {"sfpu_dest": 0, "fpu_mul": 1, "fpu_matmul": 2, "transpose": 3, "copy_only": 4}


def build(device, x, out, arm, K, outer, is_32bit=1):
    g = device.compute_with_storage_grid_size()
    core_grid = ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))]
    )
    ncores = g.x * g.y

    def cb(idx, depth):
        fmt = ttnn.CBFormatDescriptor(
            buffer_index=idx, data_format=ttnn.float32, page_size=FP32_TILE_BYTES
        )
        return ttnn.CBDescriptor(
            total_size=depth * FP32_TILE_BYTES, core_ranges=core_grid, format_descriptors=[fmt]
        )

    cbs = [cb(IN_CB, 2), cb(OUT_CB, 2), cb(SCRATCH_CB, 2)]

    reader_ct = [IN_CB, FP32_TILE_BYTES, 2]
    reader_ct.extend(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    writer_ct = [OUT_CB, FP32_TILE_BYTES]
    writer_ct.extend(ttnn.TensorAccessorArgs(out).get_compile_time_args())

    reader_rt, compute_rt, writer_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            reader_rt[cx][cy] = [x.buffer_address(), 2 * c]
            compute_rt[cx][cy] = [outer]
            writer_rt[cx][cy] = [out.buffer_address(), c]
            c += 1
    assert c == ncores

    reader = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "reader_s1b.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=reader_ct, runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "writer_s1b.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid, compile_time_args=writer_ct, runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "compute_s1b.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=core_grid,
        compile_time_args=[IN_CB, OUT_CB, SCRATCH_CB, ARMS[arm], K, is_32bit],
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True
        ),
    )
    return ttnn.ProgramDescriptor(kernels=[reader, writer, compute], semaphores=[], cbs=cbs)


def run(device, x, out, arm, K, outer, reps=5, is_32bit=1):
    pd = build(device, x, out, arm, K, outer, is_32bit)
    ttnn.generic_op([x, out], pd)          # warm: JIT compile + program cache
    ttnn.synchronize_device(device)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        r = ttnn.generic_op([x, out], pd)
        ttnn.synchronize_device(device)
        best = min(best, time.perf_counter() - t0)
    return best, ttnn.to_torch(out).clone()


def main():
    dev = ttnn.open_device(device_id=0)
    try:
        g = dev.compute_with_storage_grid_size()
        ncores = g.x * g.y
        res = {"grid": [g.x, g.y], "ncores": ncores, "l1_per_core": None}
        try:
            res["l1_per_core"] = dev.l1_size_per_core()
        except Exception as e:                                          # noqa: BLE001
            res["l1_per_core"] = f"unavailable: {e}"

        torch.manual_seed(0)
        # Values near 1.0: K back-to-back multiplies must not drift into denormals or infinity,
        # because either would change SFPU timing and turn a rate measurement into a range check.
        xt = 1.0 + 0.01 * torch.randn(1, 1, TILE * 2 * ncores, TILE, dtype=torch.float32)
        ot = torch.zeros(1, 1, TILE * ncores, TILE, dtype=torch.float32)
        x = ttnn.from_torch(xt, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.from_torch(ot, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

        OUTER = int(os.environ.get("S1B_OUTER", "20000"))
        OUT = Path(__file__).resolve().parent / "screen_s1b.json"

        # --- launch-overhead control: the K-sweep is only readable if launch is negligible --------
        res["launch_control"] = {}
        for o in (5000, 20000):
            ms, _ = run(dev, x, out, "copy_only", 0, o)
            res["launch_control"][o] = ms * 1e3
        a, b = res["launch_control"][5000], res["launch_control"][20000]
        # ns per iteration implied by the pair, with the fixed launch cost differenced out
        res["launch_control"]["per_iter_ns_differenced"] = (b - a) * 1e6 / 15000
        res["launch_control"]["per_iter_ns_naive_20k"] = b * 1e6 / 20000
        json.dump(res, open(OUT, "w"), indent=1)
        print("launch control done", flush=True)

        # --- the K sweep ---------------------------------------------------------------------------
        res["sweep"] = {}
        for arm in ("copy_only", "sfpu_dest", "fpu_mul", "fpu_matmul", "transpose"):
            res["sweep"][arm] = {}
            # sfpu_dest is the gate, so it gets the full sweep; the other arms need only enough
            # points to separate flat from linear, and every extra K is a ~25 s JIT compile.
            Ks = ((0,) if arm == "copy_only"
                  else (1, 2, 4, 8, 16, 32) if arm == "sfpu_dest"
                  else (1, 4, 16))
            for K in Ks:
                try:
                    ms, got = run(dev, x, out, arm, K, OUTER)
                    per_iter_ns = ms * 1e9 / OUTER
                    res["sweep"][arm][K] = {
                        "ms": ms * 1e3,
                        "per_iter_ns": per_iter_ns,
                        "finite": bool(torch.isfinite(got).all()),
                    }
                    print(f"{arm:11s} K={K:3d}  {per_iter_ns:8.1f} ns/iter", flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)
                except Exception as e:                                  # noqa: BLE001
                    res["sweep"][arm][K] = {"error": str(e)[:400]}
                    print(f"{arm:11s} K={K:3d}  ERROR {str(e)[:160]}", flush=True)
                    json.dump(res, open(OUT, "w"), indent=1)

        json.dump(res, open(OUT, "w"), indent=1)
        print(json.dumps(res["launch_control"], indent=1))
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
