#!/usr/bin/env python3
"""Measure an L1 bandwidth roof for the 298 aa op ledger.

Half the Pairformer block's ops never touch DRAM: input and output are both L1-sharded, so neither
DRAM roof can bind them and the ledger has no limiter to name. This measures the one that can.

Two different things get measured and they must not be conflated:

  L1 op roof   -- the fastest L1->L1 bytes/s any ttnn eltwise op reaches on this card. This is what
                  the ledger compares an L1-resident row against, because it is what a competing
                  implementation of that row could actually achieve.
  L1 hw roof   -- not measured here. Do not quote one.

Bytes are counted as every byte the op must read plus every byte it must write (binary: 3N, unary:
2N), which is the same convention the DRAM rows use.

    TT_VISIBLE_DEVICES=0 python3 perf/ledger_298/l1_roof.py --out perf/ledger_298/l1_roof_c0.json
"""
import argparse
import json
import time

import torch
import ttnn

TILE = 32


def shard_cfg(dev, rows, cols, grid_y, grid_x):
    """Block-sharded over grid_y x grid_x cores, tile aligned."""
    core_grid = ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0),
                                                 ttnn.CoreCoord(grid_x - 1, grid_y - 1))})
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.BLOCK_SHARDED, ttnn.BufferType.L1,
        ttnn.ShardSpec(core_grid, [rows // grid_y, cols // grid_x], ttnn.ShardOrientation.ROW_MAJOR))


def timed(fn, dev, iters):
    fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    dev = ttnn.open_device(device_id=0)
    gy, gx = 10, 11                      # 11x10 usable worker grid on this p150a
    runs = []
    best = {"binary": 0.0, "unary": 0.0}
    # rows/cols are tile-multiples and divisible by the grid so every core gets one shard.
    for rows, cols in [(gy * TILE, gx * TILE), (gy * TILE * 2, gx * TILE * 2),
                       (gy * TILE * 4, gx * TILE * 4), (gy * TILE * 8, gx * TILE * 8),
                       (gy * TILE * 4, gx * TILE * 16), (gy * TILE * 16, gx * TILE * 8)]:
        n = rows * cols
        mb = n * 2 / 1e6
        if mb * 3 > 170:                 # 3 live L1 tensors must fit in ~195 MB aggregate
            continue
        mc = shard_cfg(dev, rows, cols, gy, gx)
        try:
            a = ttnn.from_torch(torch.randn(rows, cols), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=mc)
            b = ttnn.from_torch(torch.randn(rows, cols), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=mc)
        except Exception as e:                                   # noqa: BLE001
            runs.append({"rows": rows, "cols": cols, "MB": round(mb, 2), "error": str(e)[:120]})
            continue

        rec = {"rows": rows, "cols": cols, "MB_per_tensor": round(mb, 3), "cores": gy * gx}
        # The output is preallocated and reused. Letting each iteration allocate and free its own
        # block-sharded L1 buffer put a flat ~125 us floor under every size in the first attempt,
        # which is allocator cost, not bandwidth.
        c = ttnn.from_torch(torch.zeros(rows, cols), layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16, memory_config=mc)
        try:
            s = timed(lambda: ttnn.add(a, b, memory_config=mc, output_tensor=c), dev, args.iters)
            rec["binary_add_us"] = round(s * 1e6, 2)
            rec["binary_GBs"] = round(3 * n * 2 / s / 1e9, 1)
            best["binary"] = max(best["binary"], rec["binary_GBs"])
        except Exception as e:                                   # noqa: BLE001
            rec["binary_error"] = str(e)[:120]
        try:
            s = timed(lambda: ttnn.mul(a, 1.0001, memory_config=mc, output_tensor=c), dev, args.iters)
            rec["unary_us"] = round(s * 1e6, 2)
            rec["unary_GBs"] = round(2 * n * 2 / s / 1e9, 1)
            best["unary"] = max(best["unary"], rec["unary_GBs"])
        except Exception as e:                                   # noqa: BLE001
            rec["unary_error"] = str(e)[:120]
        runs.append(rec)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
        ttnn.deallocate(c)
        print(rec, flush=True)

    out = {"runs": runs, "l1_op_roof_GBs": round(max(best["binary"], best["unary"]), 1),
           "binary_peak_GBs": best["binary"], "unary_peak_GBs": best["unary"],
           "note": "achievable-op roof for L1<->L1 traffic, not an SRAM hardware roof"}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=2), flush=True)
    ttnn.close_device(dev)


if __name__ == "__main__":
    main()
