#!/usr/bin/env python3
"""S1c -- is S1b's per-transaction floor a LATENCY that pipelines away, or an ISSUE cost that does not?

S1 and S1b both barriered once per assembled tile, which serialises the NoC: every iteration pays a
full round trip and no read overlaps the next iteration's. A real reader issues many reads and
barriers once. So the ~290 ns (DRAM) / ~80 ns (L1) per-read floor those screens reported may be a
property of the harness rather than of the machine, and the whole section-13 refutation of 4.2(c)
turns on which it is.

`barrier_every` assemblies share one barrier, destination cycling through that many slots so an
in-flight read is never overwritten. Latency amortises with depth; issue cost does not.
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
PAGE = 4096
TOTAL = 2048
OUTER = 5000
DEPTHS = (1, 2, 4, 8, 16, 32)
SPLITS = (1, 8, 32)
NPAGES_DRAM, NPAGES_L1 = 4096, 130


def build(dev, x, out, nreads, be, npages, offs):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    chunk = TOTAL // nreads

    def cb(i, nb, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=nb)
        return ttnn.CBDescriptor(total_size=d * nb, core_ranges=cg, format_descriptors=[f])

    rct = ([IN_CB, TOTAL, nreads, chunk, be, PAGE]
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
    # The in CB holds `be` assembly slots so no in-flight read is clobbered before its barrier.
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_s1c_pipe.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_s1_drain.cpp", [IN_CB, OUT_CB], crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(fk / "writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB, TOTAL, be + 1), cb(OUT_CB, 2048, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"page_bytes": PAGE, "total_bytes": TOTAL, "outer": OUTER, "runs": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["ncores"] = nc
        torch.manual_seed(3)
        rng = torch.Generator().manual_seed(4)
        offs = [int(torch.randint(0, PAGE, (1,), generator=rng)) for _ in range(64)]
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        srcs = {}
        for tag, npages, bt in (("dram", NPAGES_DRAM, ttnn.BufferType.DRAM),
                                ("l1", NPAGES_L1, ttnn.BufferType.L1)):
            rows = npages * PAGE // 2 // 32
            t = (0.1 * torch.randn(1, 1, rows, 32)).to(torch.bfloat16)
            mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, bt)
            srcs[tag] = (ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                         device=dev, memory_config=mc), npages)

        for tag, (x, npages) in srcs.items():
            for nreads in SPLITS:
                for be in DEPTHS:
                    key = f"{tag}/{nreads}x{TOTAL//nreads}B/be{be}"
                    try:
                        pd = build(dev, x, out, nreads, be, npages, offs)
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
                                            "nreads": nreads, "barrier_every": be,
                                            "gbs_chip": nc * TOTAL / ns, "nonzero": nz}
                        print(f"{key:26s} {ns:9.1f} ns  ({ns/nreads:7.2f} ns/read, "
                              f"{nc*TOTAL/ns:7.1f} GB/s chip)", flush=True)
                    except Exception as e:
                        res["runs"][key] = {"error": str(e)[:200]}
                        print(f"{key:26s} FAIL {str(e)[:110]}", flush=True)
                    json.dump(res, open(HERE / "s1c_pipe.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    r = res["runs"]
    print("\n--- S1c: does the floor pipeline away? ---")
    for tag in ("dram", "l1"):
        for nreads in SPLITS:
            k1 = f"{tag}/{nreads}x{TOTAL//nreads}B/be1"
            ks = [f"{tag}/{nreads}x{TOTAL//nreads}B/be{b}" for b in DEPTHS]
            vals = [(b, r[k]["ns_per_assembly"]) for b, k in zip(DEPTHS, ks)
                    if k in r and "ns_per_assembly" in r[k]]
            if not vals or k1 not in r or "ns_per_assembly" not in r[k1]:
                continue
            bb, bv = min(vals, key=lambda t: t[1])
            print(f"{tag:5s} {nreads:3d} reads: be1 {r[k1]['ns_per_assembly']:8.1f} ns -> "
                  f"best be{bb} {bv:8.1f} ns  ({r[k1]['ns_per_assembly']/bv:5.2f}x from pipelining, "
                  f"{bv/nreads:6.2f} ns/read)")
    json.dump(res, open(HERE / "s1c_pipe.json", "w"), indent=1)


main()
