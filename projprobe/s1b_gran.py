#!/usr/bin/env python3
"""S1b -- separate read-ISSUE cost from read-GRANULARITY cost, because they have opposite fixes.

S1 result: address arithmetic free (row-varying 0.855x of uniform), fragmentation ruinous (one
2048 B read 649 ns, thirty-two 64 B reads 8627 ns). Two mechanisms fit that:
  A. per-read issue cost on the dataflow RISC -> fix is fewer, larger reads, and the design's
     per-row source offsets have to go.
  B. DRAM/NoC inefficiency at small granularity -> an L1-resident source is immune, and the fix is
     the one the design already uses for the volume: make the source L1-resident and keep the
     per-row offsets as L1 addresses.
Issue cost is flat in source memory; granularity cost is not. So: fixed total bytes, split every way
from 1 read to 128, DRAM source and L1 source, plain issue and stateful issue.

Each core reads a different base page -- S1 had all 130 cores on the same page, which flatters the
bulk arm.
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
PAGE = 4096                  # source page: 4 KB, so a 64 B read is 1/64 of a page
TOTAL = 2048                 # bytes assembled per iteration = one bf16 32x32 tile
OUTER = 5000
SPLITS = (1, 2, 4, 8, 16, 32, 64)
NPAGES_DRAM = 4096           # 16 MB
NPAGES_L1 = 130              # one 4 KB page per core, L1-resident


def build(dev, x, out, nreads, mode, npages, offs):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    chunk = TOTAL // nreads

    def cb(i, d, nb):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=nb)
        return ttnn.CBDescriptor(total_size=d * nb, core_ranges=cg, format_descriptors=[f])

    rct = ([IN_CB, TOTAL, nreads, chunk, mode, PAGE]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()))
    wct = [OUT_CB, 2048] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tail = [int(o) for o in offs[:nreads]]
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [x.buffer_address(), npages, OUTER, (c * 37) % max(npages - 3, 1)] + tail
            crt[cx][cy] = [0]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    fk = HERE.parent / "fftprobe" / "s1b_kernels"
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_s1b_gran.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_s1_drain.cpp", [IN_CB, OUT_CB], crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(fk / "writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB, 2, TOTAL), cb(OUT_CB, 2, 2048)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"page_bytes": PAGE, "total_bytes": TOTAL, "outer": OUTER, "runs": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["ncores"] = nc
        torch.manual_seed(3)
        rng = torch.Generator().manual_seed(4)
        offs = [int(torch.randint(0, PAGE, (1,), generator=rng)) for _ in range(128)]
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        srcs = {}
        for tag, npages, btype in (("dram", NPAGES_DRAM, ttnn.BufferType.DRAM),
                                   ("l1", NPAGES_L1, ttnn.BufferType.L1)):
            rows = npages * PAGE // 2 // 32          # bf16, 32 wide
            t = (0.1 * torch.randn(1, 1, rows, 32)).to(torch.bfloat16)
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, btype)
            srcs[tag] = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=dev, memory_config=mc), npages)
            print(f"source {tag}: {npages} pages x {PAGE} B = {npages*PAGE/1e6:.1f} MB", flush=True)

        for tag, (x, npages) in srcs.items():
            for mname, mode in (("plain", 0), ("state", 1)):
                for nreads in SPLITS:
                    chunk = TOTAL // nreads
                    key = f"{tag}/{mname}/{nreads}x{chunk}B"
                    try:
                        pd = build(dev, x, out, nreads, mode, npages, offs)
                        ttnn.generic_op([x, out], pd)
                        ttnn.synchronize_device(dev)
                        best = float("inf")
                        for _ in range(5):
                            t0 = time.perf_counter()
                            ttnn.generic_op([x, out], pd)
                            ttnn.synchronize_device(dev)
                            best = min(best, time.perf_counter() - t0)
                        ns = best * 1e9 / OUTER
                        nz = bool(ttnn.to_torch(out).abs().sum() > 0)
                        res["runs"][key] = {"ns_per_assembly": ns, "ns_per_read": ns / nreads,
                                            "nreads": nreads, "chunk": chunk, "nonzero": nz,
                                            "gbs_chip": nc * TOTAL / ns}
                        print(f"{key:24s} {ns:9.1f} ns  ({ns/nreads:7.2f} ns/read, "
                              f"{nc*TOTAL/ns:7.1f} GB/s chip)  nz={nz}", flush=True)
                    except Exception as e:
                        res["runs"][key] = {"error": str(e)[:200]}
                        print(f"{key:24s} FAIL {str(e)[:120]}", flush=True)
                    json.dump(res, open(HERE / "s1b_gran.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    r = res["runs"]
    print("\n--- S1b: which mechanism? ---")
    for tag in ("dram", "l1"):
        k1 = f"{tag}/state/1x2048B"
        k32 = f"{tag}/state/32x64B"
        if k1 in r and "ns_per_assembly" in r[k1] and k32 in r and "ns_per_assembly" in r[k32]:
            f = r[k32]["ns_per_assembly"] / r[k1]["ns_per_assembly"]
            print(f"{tag:5s}: 32x64B / 1x2048B = {f:6.2f}x  "
                  f"({r[k1]['ns_per_assembly']:.0f} -> {r[k32]['ns_per_assembly']:.0f} ns)")
    print("Flat across dram/l1 => per-read ISSUE cost. Much smaller on l1 => GRANULARITY cost.")
    json.dump(res, open(HERE / "s1b_gran.json", "w"), indent=1)


main()
