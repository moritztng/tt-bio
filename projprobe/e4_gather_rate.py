#!/usr/bin/env python3
"""E4 -- measure the corner-gather rate for an exact-trilinear projection, and decide the route.

The pre-registered gate, written before the kernel was built (state/relion-end-to-end.md §4 E4):
RELION's coarse E-step needs 128.5 G corner gathers per iteration. To beat the 119.2 s of CPU coarse
pass by 8x that has to land in under 15 s, i.e.

    8.6 G corner-gathers/s chip-wide = 66 M/s/core over 130 cores.

A 16 B read fetches a corner PAIR (a complex fp32 voxel is 8 B and (x,y,z),(x+1,y,z) are adjacent),
so the bar in reads is 4.3 G reads/s chip = 33 M reads/s/core = 30.3 ns per read per core.

**If a single core cannot reach 66 M corner-gathers/s within 2x, the route is dead** and §3's middle
row closes by measurement rather than by a borrowed citation. Either outcome is a full result.

Arms. Chunk sweep at the real access shape, from an L1-resident 31.7 MB model:
   8 B  one corner per read      -- the pessimistic packing, if the pair trick does not hold
  16 B  a corner pair            -- THE ARM. 4 reads per output pixel.
  32 B  two pairs                -- if the cost is flat per transaction this is where to go next
  64 B  four pairs               -- the s1e_bytes anchor (43.06 ns/read at a fixed page)

`nreads` is 32 per assembly with `barrier_every` 4, S1c's measured pipelining optimum, so the number
is an issue-rate under a real reader rather than a serialised round trip.
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
BE = 4                          # S1c's measured pipelining optimum
NREADS = 32
MODEL_BYTES = 31_719_424        # the padded model, from relion-acc-backend §4.6
NPAGES = MODEL_BYTES // PAGE    # 7744 pages, 244 kB per core across 130 cores
CHUNKS = (8, 16, 32, 64)
# Control: same 16 B read, different counts per assembly. If the per-assembly cost is linear in
# nreads with an intercept near zero, the per-read number is the read and not the loop around it.
NREAD_SWEEP = (1, 2, 4, 8, 16, 32)

# The pre-registered bar, in the currency the gate is written in.
GATHERS_PER_ITER = 128.5e9
BUDGET_S = 15.0
BAR_GATHERS_PER_S_CHIP = GATHERS_PER_ITER / BUDGET_S


def build(dev, x, out, chunk, offs, strides, nreads=NREADS):
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
    tail = [int(o) for o in offs[:nreads]] + [int(s) for s in strides[:nreads]]
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rrt[cx][cy] = [x.buffer_address(), NPAGES, OUTER, c] + tail
            crt[cx][cy] = [0]
            wrt[cx][cy] = [out.buffer_address(), c]
            c += 1
    fk = HERE.parent / "fftprobe" / "s1b_kernels"
    mk = lambda p, ct, rt, cfg: ttnn.KernelDescriptor(
        kernel_source=str(p), source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt, config=cfg)
    return ttnn.ProgramDescriptor(kernels=[
        mk(KDIR / "reader_e4_gather.cpp", rct, rrt, ttnn.ReaderConfigDescriptor()),
        mk(KDIR / "compute_s1_drain.cpp", [IN_CB, OUT_CB], crt,
           ttnn.ComputeConfigDescriptor(math_fidelity=ttnn.MathFidelity.HiFi4)),
        mk(fk / "writer_s1b.cpp", wct, wrt, ttnn.WriterConfigDescriptor()),
    ], semaphores=[], cbs=[cb(IN_CB, total, BE + 1), cb(OUT_CB, 2048, 2)])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"outer": OUTER, "barrier_every": BE, "nreads": NREADS, "page_bytes": PAGE,
           "model_bytes": MODEL_BYTES, "npages": NPAGES,
           "bar": {"gathers_per_iter": GATHERS_PER_ITER, "budget_s": BUDGET_S,
                   "gathers_per_s_chip": BAR_GATHERS_PER_S_CHIP},
           "arms": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["ncores"] = nc
        res["bar"]["gathers_per_s_core"] = BAR_GATHERS_PER_S_CHIP / nc
        rng = torch.Generator().manual_seed(4)
        offs = [int(torch.randint(0, PAGE, (1,), generator=rng)) for _ in range(NREADS)]
        # Strides in the tens of kB: a projection's x-step crosses a model row, so consecutive
        # gathers land pages apart, not bytes apart.
        strides = [int(torch.randint(4096, 262144, (1,), generator=rng)) for _ in range(NREADS)]
        rows = NPAGES * PAGE // 2 // 32
        t = (0.1 * torch.randn(1, 1, rows, 32)).to(torch.bfloat16)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=mc)
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        print(f"model {MODEL_BYTES/1e6:.1f} MB L1-resident, {NPAGES} pages, "
              f"{MODEL_BYTES/nc/1024:.0f} kB/core over {nc} cores")
        print(f"bar: {BAR_GATHERS_PER_S_CHIP/1e9:.2f} G corner-gathers/s chip, "
              f"{BAR_GATHERS_PER_S_CHIP/nc/1e6:.1f} M/s/core\n")
        for chunk in CHUNKS:
            key = f"{NREADS}x{chunk}B"
            corners = chunk // 8                    # complex fp32 voxel = 8 B
            try:
                pd = build(dev, x, out, chunk, offs, strides)
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, out], pd)
                    ttnn.synchronize_device(dev)
                    best = min(best, time.perf_counter() - t0)
                # An arm whose reads the machine could have elided is not a measurement of those
                # reads. The compute kernel copies the assembled CB to the output and the writer
                # lands it, so a nonzero output is proof the gather happened.
                nz = bool(ttnn.to_torch(out).abs().sum() > 0)
                ns = best * 1e9 / OUTER             # per assembly of NREADS reads
                ns_read = ns / NREADS
                g_core = corners / ns_read * 1e9 / 1e6          # M corner-gathers/s/core
                rec = {"ns_per_assembly": ns, "ns_per_read": ns_read,
                       "corners_per_read": corners, "nonzero": nz,
                       "m_gathers_per_s_core": g_core,
                       "g_gathers_per_s_chip": g_core * nc / 1e3,
                       "gbs_chip": nc * NREADS * chunk / ns,
                       "vs_bar": g_core * 1e6 / (BAR_GATHERS_PER_S_CHIP / nc),
                       "iter_seconds": GATHERS_PER_ITER / (g_core * nc * 1e6)}
                res["arms"][key] = rec
                print(f"{key:9s} {ns_read:7.2f} ns/read  {g_core:7.1f} M gathers/s/core  "
                      f"{g_core*nc/1e3:6.2f} G/s chip  {rec['vs_bar']:.3f}x bar  "
                      f"coarse E-step {rec['iter_seconds']:8.2f} s", flush=True)
            except Exception as e:                                       # noqa: BLE001
                res["arms"][key] = {"error": str(e)[:300]}
                print(f"{key:9s} FAIL {str(e)[:200]}", flush=True)
            json.dump(res, open(HERE / "e4_gather_rate.json", "w"), indent=1)

        # The control. A flat ns/read across chunk sizes is what an issue-bound cost looks like, but
        # it is also what a loop that never issues anything looks like. Sweeping the read COUNT at a
        # fixed 16 B separates them: a real per-read cost is a straight line through the origin.
        print()
        res["control_nreads_at_16B"] = {}
        for nr in NREAD_SWEEP:
            try:
                pd = build(dev, x, out, 16, offs, strides, nreads=nr)
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, out], pd)
                    ttnn.synchronize_device(dev)
                    best = min(best, time.perf_counter() - t0)
                ns = best * 1e9 / OUTER
                res["control_nreads_at_16B"][str(nr)] = {"ns_per_assembly": ns,
                                                         "ns_per_read": ns / nr}
                print(f"control {nr:3d}x16B  {ns:8.2f} ns/assembly  {ns/nr:6.2f} ns/read",
                      flush=True)
            except Exception as e:                                       # noqa: BLE001
                res["control_nreads_at_16B"][str(nr)] = {"error": str(e)[:200]}
                print(f"control {nr:3d}x16B  FAIL {str(e)[:150]}", flush=True)
        c = res["control_nreads_at_16B"]
        if all("ns_per_assembly" in c.get(str(n), {}) for n in (1, 32)):
            # Two-point fit: slope is the marginal read, intercept is everything else in the loop.
            y1, y32 = c["1"]["ns_per_assembly"], c["32"]["ns_per_assembly"]
            slope = (y32 - y1) / 31.0
            res["control_fit"] = {"marginal_ns_per_read": slope,
                                  "loop_intercept_ns": y1 - slope}
            print(f"\nmarginal cost of one 16 B read: {slope:.2f} ns; "
                  f"loop overhead outside the reads: {y1 - slope:.2f} ns")
        json.dump(res, open(HERE / "e4_gather_rate.json", "w"), indent=1)
    finally:
        ttnn.close_device(dev)

    # THE arm is 16 B, and it is not the fastest one in the table. A trilinear cube's 8 corners are
    # adjacent in x only: (x,y,z) and (x+1,y,z) share a 16 B line, but the y and z neighbours are
    # mdlX and mdlX*mdlY voxels away. So 2 corners per read is what the interpolant actually gets on
    # the model's natural layout, and the 32 B and 64 B rows are the rate a relaid-out model WOULD
    # get, not a rate this route can claim. Reporting the max here would be the same overclaim the
    # 8.0x ceiling was.
    a = res["arms"].get(f"{NREADS}x16B", {})
    if "m_gathers_per_s_core" in a:
        res["verdict"] = {
            "arm": "32x16B, 2 corners per read, the trilinear packing on the natural layout",
            "m_gathers_per_s_core": a["m_gathers_per_s_core"],
            "bar_m_gathers_per_s_core": BAR_GATHERS_PER_S_CHIP / res["ncores"] / 1e6,
            "vs_bar": a["vs_bar"],
            "coarse_estep_seconds_1_chip": a["iter_seconds"],
            "speedup_vs_cpu_coarse_119_2s": 119.2 / a["iter_seconds"],
            "issue_bound": bool(
                max(v["ns_per_read"] for v in res["arms"].values() if "ns_per_read" in v)
                / min(v["ns_per_read"] for v in res["arms"].values() if "ns_per_read" in v) < 1.1),
            # The pre-registered kill gate: dead if a core cannot reach the bar within 2x.
            "route_dead": bool(a["vs_bar"] < 0.5),
        }
        json.dump(res, open(HERE / "e4_gather_rate.json", "w"), indent=1)
        v = res["verdict"]
        print("\n--- E4 verdict, against the gate written before the build ---")
        print(f"arm   {v['arm']}")
        print(f"rate  {v['m_gathers_per_s_core']:.1f} M gathers/s/core against a bar of "
              f"{v['bar_m_gathers_per_s_core']:.1f} -> {v['vs_bar']:.3f}x")
        print(f"cost per read is flat in transfer size: issue-bound = {v['issue_bound']}")
        print(f"coarse E-step on 1 p150 would be {v['coarse_estep_seconds_1_chip']:.2f} s "
              f"= {v['speedup_vs_cpu_coarse_119_2s']:.2f}x the 119.2 s CPU pass")
        print("ROUTE DEAD" if v["route_dead"] else "ROUTE ALIVE (within 2x of the bar)")


main()
