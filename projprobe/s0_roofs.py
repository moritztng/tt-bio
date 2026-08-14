#!/usr/bin/env python3
"""S0 -- re-verify the roofs on card 0 before anything is built on them.

Section 3.3 of the state doc inherits four per-tile-op rates from `ttnn-fft-kernel-spike`, taken on
cards 2 and 3. The section-6 budget is 0.243 us/slice of compute against a 0.312 us/slice DRAM write
floor: 1.28x of headroom, which a 20% error in the eltwise rate erases. So the rates are re-measured
here rather than inherited, per `roofline-roof-must-be-measured-not-asserted`.

Two roofs: the DRAM roof from ttnn.add on a large tensor (three tensors of traffic), and the four
per-tile-op compute rates read off the slope in K.

Kill gate (state doc section 7): matmul_tiles HiFi2 worse than 35 ns, or mul_tiles worse than 80 ns,
withdraws the section-6 budget.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
IN_CB, OUT_CB, SCRATCH_CB = 0, 16, 24
NT = 8
OUTER = 20000
KS = (4, 16)

MODES = {"matmul": 0, "mul_tiles": 1, "add_tiles": 2, "copy_tile": 3}
FID = {"LoFi": ttnn.MathFidelity.LoFi, "HiFi2": ttnn.MathFidelity.HiFi2,
       "HiFi4": ttnn.MathFidelity.HiFi4}
DT = {"fp32": (ttnn.float32, torch.float32, 4), "bf16": (ttnn.bfloat16, torch.bfloat16, 2)}

# The arms worth spending time on: the two rates the budget rests on, plus the fidelity sweep on the
# matmul (because fp32-costs-nothing-over-bf16 is the claim that makes fp32 backprojection cheap).
ARMS = [
    ("matmul", "fp32", "HiFi2"), ("matmul", "fp32", "HiFi4"), ("matmul", "fp32", "LoFi"),
    ("matmul", "bf16", "HiFi2"), ("matmul", "bf16", "HiFi4"),
    ("mul_tiles", "fp32", "HiFi4"), ("mul_tiles", "bf16", "HiFi4"),
    ("add_tiles", "fp32", "HiFi4"), ("add_tiles", "bf16", "HiFi4"),
    ("copy_tile", "fp32", "HiFi4"),
]


def build(dev, x, out, K, mode, fid, dt):
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
    fk = Path(HERE.parent / "fftprobe" / "s1b_kernels")
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(fk / "reader_s1b.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(fk / "writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
        mk(KDIR / "compute_s0.cpp", [IN_CB, SCRATCH_CB, OUT_CB, K, NT, MODES[mode]], crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=FID[fid], fp32_dest_acc_en=(dt == "fp32"))),
    ], semaphores=[], cbs=[cb(IN_CB, NT), cb(SCRATCH_CB, 2), cb(OUT_CB, 2)])


def dram_roof(dev, res):
    """ttnn.add on 8192x8192: two reads and one write, so 3 tensors of traffic."""
    for dt, (tdt, tor, nb) in DT.items():
        n = 8192
        a = ttnn.from_torch(torch.randn(1, 1, n, n).to(tor), dtype=tdt, layout=ttnn.TILE_LAYOUT,
                            device=dev)
        b = ttnn.from_torch(torch.randn(1, 1, n, n).to(tor), dtype=tdt, layout=ttnn.TILE_LAYOUT,
                            device=dev)
        ttnn.add(a, b)
        ttnn.synchronize_device(dev)
        best = float("inf")
        for _ in range(5):
            t0 = time.perf_counter()
            c = ttnn.add(a, b)
            ttnn.synchronize_device(dev)
            best = min(best, time.perf_counter() - t0)
            ttnn.deallocate(c)
        gbs = 3 * n * n * nb / best / 1e9
        res["dram_roof"][dt] = {"ms": best * 1e3, "gbs": gbs}
        print(f"DRAM roof {dt:5s} {best*1e3:7.3f} ms  {gbs:7.1f} GB/s", flush=True)
        ttnn.deallocate(a); ttnn.deallocate(b)


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"outer": OUTER, "ks": list(KS), "nt": NT, "dram_roof": {}, "tile_ops": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        n = g.x * g.y
        res["ncores"] = n
        print(f"grid {g.x}x{g.y} = {n} cores", flush=True)
        dram_roof(dev, res)

        torch.manual_seed(0)
        tensors = {}
        for dt, (tdt, tor, _) in DT.items():
            xt = (0.1 * torch.randn(1, 1, 32 * NT * n, 32)).to(tor)
            tensors[dt] = (
                ttnn.from_torch(xt, dtype=tdt, layout=ttnn.TILE_LAYOUT, device=dev),
                ttnn.from_torch(torch.zeros(1, 1, 32 * n, 32).to(tor), dtype=tdt,
                                layout=ttnn.TILE_LAYOUT, device=dev))
        for mode, dt, fid in ARMS:
            x, out = tensors[dt]
            ts = {}
            for K in KS:
                pd = build(dev, x, out, K, mode, fid, dt)
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
            fixed = ts[KS[0]] - KS[0] * slope
            key = f"{mode}/{dt}/{fid}"
            res["tile_ops"][key] = {"ns_per_tile_op": slope, "fixed_ns_per_iter": fixed,
                                    "per_iter_ns": {str(k): v for k, v in ts.items()}}
            print(f"{key:26s} {slope:7.2f} ns/tile-op   (fixed {fixed:6.1f} ns/iter)", flush=True)
            json.dump(res, open(HERE / "s0_roofs.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    # --- kill gate, stated in the state doc section 7 -------------------------------------------
    mm = res["tile_ops"].get("matmul/fp32/HiFi2", {}).get("ns_per_tile_op")
    ml = res["tile_ops"].get("mul_tiles/fp32/HiFi4", {}).get("ns_per_tile_op")
    print("\n--- S0 kill gate ---")
    print(f"matmul HiFi2 {mm:.2f} ns  (gate: < 35)  -> {'PASS' if mm < 35 else 'FAIL'}")
    print(f"mul_tiles    {ml:.2f} ns  (gate: < 80)  -> {'PASS' if ml < 80 else 'FAIL'}")
    res["gate"] = {"matmul_hifi2_ns": mm, "mul_tiles_ns": ml,
                   "pass": bool(mm < 35 and ml < 80)}
    json.dump(res, open(HERE / "s0_roofs.json", "w"), indent=1)


main()
