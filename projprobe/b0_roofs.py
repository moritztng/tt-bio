#!/usr/bin/env python3
"""B0 -- the DRAM READ, WRITE and READ-MODIFY-WRITE roofs, measured separately on this card.

The brief for backprojection is explicit that the floor must come from the write side and that
nothing may be inherited, including from the projection pass. Every roof in this program so far is
`ttnn.add` on an 8192x8192 tensor: two reads and one write, 420.2 GB/s. That number cannot be used
as a write roof, because it is not one.

Three arms, all on 130 cores, all on the 2048 B tile page the built kernels actually move, plus the
128 B scattered row the adjoint's writer issues:

  read  -- noc_async_read  DRAM -> L1, `be` pages in flight per barrier
  write -- noc_async_write L1 -> DRAM, same depth
  rmw   -- read the group, barrier, write the group back, barrier

Each core owns a disjoint page range, so the arms measure the memory system rather than contention
between cores for the same page.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
KDIR = HERE / "kernels"
CB = 0
PAGE = 2048
OUTER = 4096
MODES = {"read": 0, "write": 1, "rmw": 2}
DEPTHS = (1, 4, 16)
XACTS = (2048, 128)
MB = 512  # DRAM buffer, MiB-ish: PAGES * PAGE


def build(dev, x, mode, xact, be, page0_of_core, npage):
    g = dev.compute_with_storage_grid_size()
    cg = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
    fmt = ttnn.CBFormatDescriptor(buffer_index=CB, data_format=ttnn.bfloat16, page_size=PAGE)
    cbd = ttnn.CBDescriptor(total_size=be * PAGE, core_ranges=cg, format_descriptors=[fmt])
    ct = [CB, PAGE, xact, mode, be] + list(ttnn.TensorAccessorArgs(x).get_compile_time_args())
    rt = ttnn.RuntimeArgs()
    c = 0
    for cy in range(g.y):
        for cx in range(g.x):
            rt[cx][cy] = [x.buffer_address(), page0_of_core[c], npage, OUTER, 0x3F803F80]
            c += 1
    k = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "roof_rw.cpp"),
        source_type=ttnn.KernelDescriptor.SourceType.FILE_PATH,
        core_ranges=cg, compile_time_args=ct, runtime_args=rt,
        config=ttnn.ReaderConfigDescriptor())
    return ttnn.ProgramDescriptor(kernels=[k], semaphores=[], cbs=[cbd])


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"page_bytes": PAGE, "outer": OUTER, "roofs": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["ncores"] = nc
        print(f"grid {g.x}x{g.y} = {nc} cores", flush=True)

        npage_per_core = 512
        pages = nc * npage_per_core
        rows = pages * PAGE // 2 // 32
        torch.manual_seed(0)
        t = (0.1 * torch.randn(1, 1, rows, 32)).to(torch.bfloat16)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.DRAM)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=mc)
        print(f"buffer {pages} pages x {PAGE} B = {pages*PAGE/2**20:.0f} MiB", flush=True)
        page0 = [c * npage_per_core for c in range(nc)]

        for xact in XACTS:
            for name, mode in MODES.items():
                for be in DEPTHS:
                    key = f"{name}/{xact}B/be{be}"
                    try:
                        pd = build(dev, x, mode, xact, be, page0, npage_per_core)
                        ttnn.generic_op([x], pd)
                        ttnn.synchronize_device(dev)
                        best = float("inf")
                        for _ in range(5):
                            t0 = time.perf_counter()
                            ttnn.generic_op([x], pd)
                            ttnn.synchronize_device(dev)
                            best = min(best, time.perf_counter() - t0)
                        # traffic: rmw crosses DRAM twice per page.
                        mult = 2 if mode == 2 else 1
                        by = nc * OUTER * PAGE * mult
                        ns_page = best * 1e9 / OUTER
                        res["roofs"][key] = {
                            "ms": best * 1e3, "gbs": by / best / 1e9,
                            "ns_per_page": ns_page, "ns_per_xact": ns_page / (PAGE // xact),
                            "xact_bytes": xact, "barrier_every": be, "mode": name}
                        print(f"{key:22s} {best*1e3:8.3f} ms  {by/best/1e9:7.1f} GB/s  "
                              f"{ns_page:8.1f} ns/page  {ns_page/(PAGE//xact):7.2f} ns/xact",
                              flush=True)
                    except Exception as e:
                        res["roofs"][key] = {"error": str(e)[:220]}
                        print(f"{key:22s} FAIL {str(e)[:140]}", flush=True)
                    json.dump(res, open(HERE / "b0_roofs.json", "w"), indent=1)

        # The write arms wrote 0x3F803F80 = two bf16 1.0 into every page they touched. If the
        # buffer still looks like the random init, the writes did not land and the roof is a lie.
        back = ttnn.to_torch(x).float()
        frac_ones = float((back == 1.0).float().mean())
        res["write_landed_frac_ones"] = frac_ones
        print(f"\nwrite check: {frac_ones*100:.1f}% of the buffer is exactly 1.0", flush=True)
    finally:
        ttnn.close_device(dev)

    r = res["roofs"]
    print("\n--- B0: the three roofs, best over barrier depth, 2048 B pages ---")
    for name in MODES:
        vals = [(be, r[f"{name}/2048B/be{be}"]["gbs"]) for be in DEPTHS
                if "gbs" in r.get(f"{name}/2048B/be{be}", {})]
        if vals:
            bb, bv = max(vals, key=lambda t: t[1])
            print(f"{name:6s} {bv:7.1f} GB/s  (best at be{bb})")
    json.dump(res, open(HERE / "b0_roofs.json", "w"), indent=1)


main()
