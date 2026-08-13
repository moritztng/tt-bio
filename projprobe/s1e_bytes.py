#!/usr/bin/env python3
"""S1e -- does a bigger read per row cost anything? This decides a factor of 2 in the reader budget.

Section 6 keeps the real and imaginary components of the Fourier slice in SEPARATE tiles (its stage-2
cost is "2 matmuls, one per real component"). So a complex output tile is two physical tiles, and the
question is whether the reader pays for that twice.

S1b and S1c both showed the cost per read is FLAT in transfer size below 512 B -- per transaction, not
per byte. If that holds, then interleaving real and imaginary per source row so one read fetches
128 B instead of 64 B is FREE, and a complex output tile costs one 32-read assembly rather than two.
If it does not hold, the section-14.3 reader budget doubles and the predicted rate halves.

Arms, all from an L1-resident source at the depth-4 optimum S1c found:
  32 x  64 B = 2048 B   one component
  32 x 128 B = 4096 B   both components interleaved -- the arm that matters
  32 x 256 B = 8192 B   the knee, to show where flatness ends
  64 x  64 B = 4096 B   both components as two separate 32-read assemblies, for the comparison
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
OUTER = 5000
BE = 4                       # S1c's optimum depth
NPAGES = 130
ARMS = ((32, 64), (32, 128), (32, 256), (64, 64))


def build(dev, x, out, nreads, chunk, offs):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    total = nreads * chunk

    def cb(i, nb, d):
        f = ttnn.CBFormatDescriptor(buffer_index=i, data_format=ttnn.bfloat16, page_size=nb)
        return ttnn.CBDescriptor(total_size=d * nb, core_ranges=cg, format_descriptors=[f])

    rct = ([IN_CB, total, nreads, chunk, BE, PAGE]
           + list(ttnn.TensorAccessorArgs(x).get_compile_time_args()))
    wct = [OUT_CB, 2048] + list(ttnn.TensorAccessorArgs(out).get_compile_time_args())
    rrt, crt, wrt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    tail = [int(o) for o in offs[:nreads]]
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [x.buffer_address(), NPAGES, OUTER, (c * 37) % (NPAGES - 3)] + tail
            crt[cx][cy] = [0]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    fk = HERE.parent / "fftprobe" / "s1b_kernels"
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_s1c_pipe.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_s1_drain.cpp", [IN_CB, OUT_CB], crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(fk / "writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB, total, BE + 1), cb(OUT_CB, 2048, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"barrier_every": BE, "outer": OUTER, "arms": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["ncores"] = nc
        torch.manual_seed(9)
        rng = torch.Generator().manual_seed(10)
        offs = [int(torch.randint(0, PAGE, (1,), generator=rng)) for _ in range(64)]
        rows = NPAGES * PAGE // 2 // 32
        t = (0.1 * torch.randn(1, 1, rows, 32)).to(torch.bfloat16)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=mc)
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        for nreads, chunk in ARMS:
            key = f"{nreads}x{chunk}B"
            try:
                pd = build(dev, x, out, nreads, chunk, offs)
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, out], pd)
                    ttnn.synchronize_device(dev)
                    best = min(best, time.perf_counter() - t0)
                ns = best * 1e9 / OUTER
                res["arms"][key] = {"ns_per_assembly": ns, "total_bytes": nreads * chunk,
                                    "ns_per_read": ns / nreads,
                                    "gbs_chip": nc * nreads * chunk / ns}
                print(f"{key:10s} {nreads*chunk:6d} B  {ns:8.1f} ns  "
                      f"({ns/nreads:6.2f} ns/read, {nc*nreads*chunk/ns:7.1f} GB/s chip)", flush=True)
            except Exception as e:
                res["arms"][key] = {"error": str(e)[:200]}
                print(f"{key:10s} FAIL {str(e)[:120]}", flush=True)
            json.dump(res, open(HERE / "s1e_bytes.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    a = res["arms"]
    print("\n--- S1e: is the second complex component free? ---")
    if all(k in a and "ns_per_assembly" in a[k] for k in ("32x64B", "32x128B", "64x64B")):
        one, both_i, both_s = (a["32x64B"]["ns_per_assembly"], a["32x128B"]["ns_per_assembly"],
                               a["64x64B"]["ns_per_assembly"])
        print(f"one component        32x 64B  {one:8.1f} ns")
        print(f"both, interleaved    32x128B  {both_i:8.1f} ns  = {both_i/one:.3f}x")
        print(f"both, two assemblies 64x 64B  {both_s:8.1f} ns  = {both_s/one:.3f}x")
        print(f"interleaving saves {both_s/both_i:.2f}x over separate assemblies")
        res["verdict"] = {"one_component_ns": one, "interleaved_ns": both_i,
                          "separate_ns": both_s, "interleave_gain": both_s / both_i,
                          "second_component_is_free": bool(both_i < 1.15 * one)}
        json.dump(res, open(HERE / "s1e_bytes.json", "w"), indent=1)


main()
