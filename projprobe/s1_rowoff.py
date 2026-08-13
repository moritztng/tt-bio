#!/usr/bin/env python3
"""S1 -- does the per-row integer offset stay free? The screen that can kill the design.

State doc section 7. Section 4.2(c) puts every bit of an affine warp's per-row variation into a
per-row byte offset in the reader. If a row-varying read costs materially more than a uniform one,
that assumption is a 32-way scalar loop in disguise and section 6's prediction has to be rebuilt
against the rejected 4.6 ns/output variant.

Two things are reported, because the stated ratio gate is not the whole question:
  * the RATIO of row-varying to uniform, which is the gate the plan wrote (kill at > 1.3x);
  * the ABSOLUTE ns per tile-assembly against the per-core budget, which is what actually decides
    whether the reader fits. Box 256: 25,736 output points = 25.1 tiles per slice over 130 cores =
    0.193 output tiles per core per slice, in a 0.312 us DRAM-write budget, so 1.61 us per output
    tile. At roughly 3 tile-assemblies per output tile (stage 2a, 2b, and stage 1's share) the
    reader has about 540 ns per assembly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
IN_CB, OUT_CB = 0, 16
TB = 32 * 32 * 2            # bf16 tile
NPAGES = 4096               # 8 MB of DRAM source, so reads are not served from one hot page
OUTER = 20000

ARMS = [("BULK", 0, 32), ("UNIF32", 1, 32), ("VARY32", 2, 32),
        ("STATE32", 3, 32), ("STATE64", 4, 64)]

# Per-core budget derived in the docstring; carried here so the screen reports its own verdict.
BUDGET_NS = 540.0


def build(dev, x, out, mode, nrows, offs):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def cb(i, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=TB)
        return ttnn.CBDescriptor(total_size=d * TB, core_ranges=cg, format_descriptors=[f])

    rct = [IN_CB, TB, mode, nrows] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    wct = [OUT_CB, TB] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tail = []
    for r in range(nrows):
        tail += [int(offs[r][0]), int(offs[r][1])]
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [x.buffer_address(), NPAGES, OUTER] + tail
            crt[cx][cy] = [0]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    fk = HERE.parent / "fftprobe" / "s1b_kernels"
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_s1_rowoff.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_s1_drain.cpp", [IN_CB, OUT_CB], crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(fk / "writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB, 2), cb(OUT_CB, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"outer": OUTER, "tile_bytes": TB, "npages": NPAGES, "budget_ns": BUDGET_NS, "arms": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        res["ncores"] = g.x * g.y
        torch.manual_seed(1)
        xt = (0.1 * torch.randn(1, 1, 32 * NPAGES, 32)).to(torch.bfloat16)
        x = ttnn.from_torch(xt, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * g.x * g.y, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        # A shear's row offsets: monotone in the row, spanning a couple of pages, never repeating.
        rng = torch.Generator().manual_seed(2)
        offs64 = [(r // 8, int(torch.randint(0, TB // 2, (1,), generator=rng)) & ~0x1F)
                  for r in range(64)]
        for name, mode, nrows in ARMS:
            pd = build(dev, x, out, mode, nrows, offs64)
            ttnn.generic_op([x, out], pd)
            ttnn.synchronize_device(dev)
            best = float("inf")
            for _ in range(5):
                t0 = time.perf_counter()
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = min(best, time.perf_counter() - t0)
            got = ttnn.to_torch(out)
            ns = best * 1e9 / OUTER
            nonzero = bool(got.abs().sum() > 0)
            res["arms"][name] = {"ns_per_tile_assembly": ns, "nreads": nrows,
                                 "ns_per_read": ns / nrows, "nonzero": nonzero,
                                 "gbs_chip": res["ncores"] * TB / ns}
            print(f"{name:9s} {nrows:3d} reads  {ns:8.1f} ns/tile-assembly "
                  f"({ns/nrows:6.2f} ns/read)  nonzero={nonzero}", flush=True)
            json.dump(res, open(HERE / "s1_rowoff.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    a = res["arms"]
    print("\n--- S1 gates ---")
    for k in ("VARY32", "STATE32", "STATE64"):
        if k in a and "UNIF32" in a:
            rat = a[k]["ns_per_tile_assembly"] / a["UNIF32"]["ns_per_tile_assembly"]
            print(f"{k} / UNIF32 = {rat:.3f}x   (plan gate: < 1.3)  "
                  f"-> {'PASS' if rat < 1.3 else 'FAIL'}")
            a[k]["ratio_to_uniform"] = rat
    best = min((a[k]["ns_per_tile_assembly"] for k in ("VARY32", "STATE32", "STATE64") if k in a),
               default=None)
    if best is not None:
        print(f"cheapest row-varying assembly {best:.1f} ns  vs {BUDGET_NS:.0f} ns budget  "
              f"-> {'PASS' if best < BUDGET_NS else 'FAIL'}")
        res["verdict"] = {"best_varying_ns": best, "budget_ns": BUDGET_NS,
                          "fits_budget": bool(best < BUDGET_NS)}
    json.dump(res, open(HERE / "s1_rowoff.json", "w"), indent=1)


main()
