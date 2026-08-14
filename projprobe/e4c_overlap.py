#!/usr/bin/env python3
"""E4c -- does the trilinear blend hide under the gather?

E4b put the coarse E-step gather at 9.37 s on one p150 (18.96 ns per 16 B read, two dataflow RISCs).
That is a floor for the whole coarse pass only if the arithmetic rides along for free. The math is
tile-wise once the reader has put each gathered corner pair in the right slot, and it can:
`noc_async_read` picks its own L1 destination, so the CB comes out already tile-shaped and no
transpose or shuffle is needed. One assembly of 32 x 16 B is 8 pixels' worth of corner pairs, and a
pixel needs about 8 weighted accumulates per complex component, so the interesting neighbourhood is
tens of tile ops per assembly.

Imports the settled harness rather than re-running its arms: the chunk sweep, the read-count control
and the dual-RISC arm are already measured three times over and nothing here changes them.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import ttnn

import e4_gather_rate as E

OPS = (0, 4, 8, 16, 32, 64, 128)


def main():
    dev = ttnn.open_device(device_id=0)
    res = {"outer": E.OUTER, "nreads": E.NREADS, "chunk": 16, "ops_sweep": {}}
    try:
        g = dev.compute_with_storage_grid_size()
        nc = g.x * g.y
        res["ncores"] = nc
        rng = torch.Generator().manual_seed(4)
        offs = [int(torch.randint(0, E.PAGE, (1,), generator=rng)) for _ in range(E.NREADS)]
        strides = [int(torch.randint(4096, 262144, (1,), generator=rng)) for _ in range(E.NREADS)]
        rows = E.NPAGES * E.PAGE // 2 // 32
        t = (0.1 * torch.randn(1, 1, rows, 32)).to(torch.bfloat16)
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
        x = ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                            memory_config=mc)
        out = ttnn.from_torch(torch.zeros(1, 1, 32 * nc, 32).to(torch.bfloat16),
                              dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
        base = None
        for ops in OPS:
            try:
                pd = E.build_dual(dev, x, out, 16, offs, strides, ops=ops)
                ttnn.generic_op([x, out], pd)
                ttnn.synchronize_device(dev)
                best = float("inf")
                for _ in range(5):
                    t0 = time.perf_counter()
                    ttnn.generic_op([x, out], pd)
                    ttnn.synchronize_device(dev)
                    best = min(best, time.perf_counter() - t0)
                nz = bool(ttnn.to_torch(out).abs().sum() > 0)
                ns = best * 1e9 / E.OUTER
                if base is None:
                    base = ns
                rec = {"ns_per_assembly_pair": ns, "vs_gather_only": ns / base, "nonzero": nz,
                       "tile_ops": ops,
                       "coarse_estep_s_1_chip": E.GATHERS_PER_ITER
                       / (2 * 64 / (ns * 1e-9) * nc)}
                res["ops_sweep"][str(ops)] = rec
                print(f"ops={ops:4d}  {ns:8.2f} ns/assembly-pair  {ns/base:.3f}x gather-only  "
                      f"coarse E-step {rec['coarse_estep_s_1_chip']:6.2f} s", flush=True)
            except Exception as e:                                       # noqa: BLE001
                res["ops_sweep"][str(ops)] = {"error": str(e)[:300]}
                print(f"ops={ops:4d}  FAIL {str(e)[:200]}", flush=True)
            json.dump(res, open(Path(__file__).resolve().parent / "e4c_overlap.json", "w"),
                      indent=1)
    finally:
        ttnn.close_device(dev)

    # What the real kernel needs, in the same currency, so "free up to 16" means something.
    # One assembly-pair is 64 reads x 16 B = 128 corners = 16 output pixels (8 corners each).
    # A tile op processes 1024 elements, so per output pixel the arithmetic is:
    #   trilinear blend  8 weighted accumulates x 2 complex components      = 16 element-ops
    #   the compare      9 translations x (2 sub, 2 mul, 1 add, 1 scale)    = 54 element-ops
    # 70 element-ops per pixel x 16 pixels / 1024 elements per tile op = 1.09 tile ops per
    # assembly-pair. That is the number the sweep has to clear, and it is measured at HiFi4, the
    # slowest fidelity, so the margin is conservative.
    NEEDED = 16 * (8 * 2 + 9 * 6) / 1024.0
    s = res["ops_sweep"]
    free = [int(k) for k, v in s.items()
            if "vs_gather_only" in v and v["vs_gather_only"] < 1.05]
    if free:
        res["verdict"] = {
            "free_up_to_tile_ops_per_assembly_pair": max(free),
            "needed_tile_ops_per_assembly_pair": NEEDED,
            "headroom": max(free) / NEEDED,
            "gather_only_ns": s["0"]["ns_per_assembly_pair"],
            "blend_hides": bool(max(free) > NEEDED),
        }
        json.dump(res, open(Path(__file__).resolve().parent / "e4c_overlap.json", "w"), indent=1)
        v = res["verdict"]
        print(f"\nfree up to {v['free_up_to_tile_ops_per_assembly_pair']} tile ops per "
              f"assembly-pair; the blend and the compare need {NEEDED:.2f} "
              f"-> {v['headroom']:.1f}x headroom")
        print("the blend hides under the gather, so the gather is the binding term"
              if v["blend_hides"]
              else "the blend does NOT hide: the coarse-pass floor is above the gather")


main()
