#!/usr/bin/env python3
"""S1d -- assemble a tile from the core's OWN L1 with RISC copies. The last reader option.

S1c's L1 arm read an INTERLEAVED L1 tensor, so nearly every read was a remote L1 read over the NoC
and paid NoC latency; it pipelined to 1218 ns for a 32 x 64 B assembly. The design's strip is resident
in the core's own L1, which needs no NoC transaction at all. This measures the RISC doing it directly:
one bulk strip read outside the timed loop, then per-output-tile assembly out of that strip.

NOOP subtracts the loop, LOCAL_BULK moves the same 2048 B in one contiguous run, so LOCAL_VARY is
charged for its fragmentation and not for its bytes.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
SRC_CB, DST_CB, OUT_CB = 0, 8, 16
PAGE = 2048
STRIP_PAGES = 16          # 32 KB strip resident in own L1
NROWS, CHUNK = 32, 64
OUTER = 5000
ARMS = (("LOCAL_VARY", 0), ("LOCAL_BULK", 1), ("NOOP", 2))


def build(dev, x, out, mode, offs):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])

    def cb(i, nb, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=nb)
        return ttnn.CBDescriptor(total_size=d * nb, core_ranges=cg, format_descriptors=[f])

    rct = ([SRC_CB, DST_CB, STRIP_PAGES, PAGE, NROWS, CHUNK, mode]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()))
    wct = [OUT_CB, 2048] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tail = [int(o) for o in offs[:NROWS]]
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [x.buffer_address(), (c * STRIP_PAGES) % 2048, OUTER] + tail
            crt[cx][cy] = [0]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    fk = HERE.parent / "fftprobe" / "s1b_kernels"
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_s1d_local.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_s1_drain.cpp", [DST_CB, OUT_CB], crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(fk / "writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(SRC_CB, PAGE, STRIP_PAGES), cb(DST_CB, 2048, 2), cb(OUT_CB, 2048, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"strip_bytes": STRIP_PAGES * PAGE, "nrows": NROWS, "chunk": CHUNK, "outer": OUTER,
           "arms": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["ncores"] = nc
        torch.manual_seed(5)
        rng = torch.Generator().manual_seed(6)
        offs = [int(torch.randint(0, STRIP_PAGES * PAGE - CHUNK, (1,), generator=rng))
                for _ in range(NROWS)]
        t = (0.1 * torch.randn(1, 1, 32 * 4096, 32)).to(torch.bfloat16)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for name, mode in ARMS:
            try:
                pd = build(dev, x, out, mode, offs)
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, out], pd)
                    ttnn.synchronize_device(dev)
                    best = min(best, time.perf_counter() - t0)
                ns = best * 1e9 / OUTER
                res["arms"][name] = {"ns_per_assembly": ns}
                print(f"{name:12s} {ns:9.1f} ns per 2048 B assembly", flush=True)
            except Exception as e:
                res["arms"][name] = {"error": str(e)[:200]}
                print(f"{name:12s} FAIL {str(e)[:130]}", flush=True)
            json.dump(res, open(HERE / "s1d_local.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    a = res["arms"]
    if all(k in a and "ns_per_assembly" in a[k] for k, _ in ARMS):
        noop = a["NOOP"]["ns_per_assembly"]
        v = a["LOCAL_VARY"]["ns_per_assembly"] - noop
        b = a["LOCAL_BULK"]["ns_per_assembly"] - noop
        print(f"\n--- S1d, loop subtracted ({noop:.1f} ns) ---")
        print(f"LOCAL_VARY  {v:8.1f} ns   LOCAL_BULK {b:8.1f} ns   "
              f"fragmentation costs {v/b:.2f}x")
        print(f"vs S1c's best NoC assembly of 1218.5 ns: "
              f"{'LOCAL wins ' + format(1218.5/v, '.2f') + 'x' if v < 1218.5 else 'NoC wins'}")
        res["verdict"] = {"local_vary_ns": v, "local_bulk_ns": b, "noop_ns": noop,
                          "frag_factor": v / b, "vs_noc_1218ns": 1218.5 / v}
        json.dump(res, open(HERE / "s1d_local.json", "w"), indent=1)


main()
