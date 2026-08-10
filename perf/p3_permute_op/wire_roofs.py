#!/usr/bin/env python3
"""Roofs for the wired channel move, measured on THIS card in THIS pass (charter §4.1).

The op's own shape is the trimul chunk `[1, 320, 320, 32]` bf16 = 6.5536 MB one way, so the copy
floors have to be taken at that size and not at the 48.82 MB pair-tensor size X4 used for a
different op. Nothing here is inherited.
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn
from tt_bio import reblock_permute as RP

N = 320
BYTES = N * N * 32 * 2


def timeit(device, fn, reps=15, warmup=3):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="perf/p3_permute_op/wire_roofs.json")
    a = ap.parse_args()
    RP.set_enabled(True)
    device = ttnn.open_device(device_id=0)
    g = device.compute_with_storage_grid_size()
    per_core_l1 = int(ttnn.get_max_worker_l1_unreserved_size())
    R = {"grid": [g.x, g.y], "cores": g.x * g.y, "per_core_l1_unreserved_B": per_core_l1,
         "shape": [1, N, N, 32], "bytes_one_way": BYTES, "clone_floors": [], "op": {}}

    ref = torch.randn(1, N, N, 32, dtype=torch.bfloat16)
    for src in ("l1", "dram"):
        smc = ttnn.L1_MEMORY_CONFIG if src == "l1" else ttnn.DRAM_MEMORY_CONFIG
        x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                            memory_config=smc)
        for dst in ("l1", "dram"):
            dmc = ttnn.L1_MEMORY_CONFIG if dst == "l1" else ttnn.DRAM_MEMORY_CONFIG

            def c(x=x, dmc=dmc):
                o = ttnn.clone(x, memory_config=dmc)
                ttnn.deallocate(o)

            us = timeit(device, c)
            R["clone_floors"].append({"src": src, "dst": dst, "us": round(us, 2),
                                      "gb_s_one_way": round(BYTES / us / 1e3, 1)})
            print(R["clone_floors"][-1], flush=True)
        ttnn.deallocate(x)

    # The op itself, at the production buffer type (L1), with the engaged-core count read out of the
    # CoreRangeSet the work split actually returned rather than assumed.
    mc = ttnn.L1_MEMORY_CONFIG
    x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                        memory_config=mc)
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([1, 32, N, N]), ttnn.bfloat16,
                                         ttnn.TILE_LAYOUT, device, mc)
    entry = RP._prepare(x, out, device)
    cg = entry["core_grid"]
    ncores = sum((cr.end.x - cr.start.x + 1) * (cr.end.y - cr.start.y + 1) for cr in cg.ranges())
    ttnn.deallocate(out)

    def wired(x=x, mc=mc):
        o = RP.reblock_permute(x, mc, device)
        ttnn.deallocate(o)

    def base(x=x, mc=mc):
        o = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
        ttnn.deallocate(o)

    us_w, us_b = timeit(device, wired), timeit(device, base)
    ntiles = N * N * 32 // 1024
    R["op"] = {"cores_engaged": ncores, "of_grid": g.x * g.y, "tiles": ntiles,
               "wired_us": round(us_w, 2), "ttnn_permute_us": round(us_b, 2),
               "wired_gb_s_one_way": round(BYTES / us_w / 1e3, 1),
               "base_gb_s_one_way": round(BYTES / us_b / 1e3, 1),
               "us_per_tile_per_engaged_core": round(us_w / ntiles * ncores, 3),
               "flops": 0, "arithmetic_intensity_flop_per_byte": 0.0}
    print(R["op"], flush=True)
    ttnn.deallocate(x)
    ttnn.close_device(device)
    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    print(json.dumps(R, indent=2))


if __name__ == "__main__":
    main()
