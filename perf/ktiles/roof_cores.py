#!/usr/bin/env python3
"""DRAM read bandwidth as a function of engaged core count, on THIS card.

Every "% of roof" this sprint quotes for a partially-occupied op sits between two numbers
nobody measured: the whole-grid roof (395.3 GB/s on card 2, all 130 cores pulling) and a naive
proportional roof (395.3 * k/130), which for the DiT AV class predicts a time SLOWER than the
one measured. The truth is in between. This measures it.

Method matches perf/ledger_298/roofs_card.py's read leg -- a DRAM-interleaved tensor cloned to
L1, so DRAM sees reads only -- with one change: the L1 destination is HEIGHT_SHARDED over a
CoreRangeSet of exactly k cores, so exactly k cores issue the reads. Bytes per core are held
constant, so total bytes scale with k and every point is the same per-core working set.
Every timed region synchronises immediately before the clock starts and before it stops.
"""
import argparse, json, statistics as st, time

import torch
import ttnn

from tt_bio.tenstorrent import get_device

TILE = 32


def core_set(k, grid_x):
    """Row-major CoreRangeSet of exactly k cores on a grid_x-wide grid."""
    full, rem = divmod(k, grid_x)
    ranges = []
    if full:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(grid_x - 1, full - 1)))
    if rem:
        ranges.append(ttnn.CoreRange(ttnn.CoreCoord(0, full), ttnn.CoreCoord(rem - 1, full)))
    return ttnn.CoreRangeSet(set(ranges))


def timed(fn, dev, warm=3, pipe=4, reps=7):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--kb-per-core", type=int, default=512)
    ap.add_argument("--cols", type=int, default=4096)
    a = ap.parse_args()

    dev = get_device()
    dg = dev.compute_with_storage_grid_size()
    grid_x, grid_y = dg.x, dg.y
    cores_max = grid_x * grid_y

    # rows per core, tile aligned: kb_per_core * 1024 bytes / (cols * 2 bytes per bf16)
    rows_per_core = int(a.kb_per_core * 1024 / (a.cols * 2))
    assert rows_per_core % TILE == 0, f"{rows_per_core} rows/core is not tile aligned"

    ks = [k for k in (1, 2, 4, 8, 16, 32, 64, 96, cores_max) if k <= cores_max]
    res = {"card_grid": f"{grid_x}x{grid_y}", "cores_max": cores_max,
           "kb_per_core": a.kb_per_core, "cols": a.cols, "rows_per_core": rows_per_core,
           "ttnn": getattr(ttnn, "__version__", "?"), "points": []}
    print(f"grid {grid_x}x{grid_y} = {cores_max} cores, {rows_per_core} rows/core "
          f"({a.kb_per_core} KB), {a.cols} cols", flush=True)

    for k in ks:
        rows = rows_per_core * k
        nbytes = rows * a.cols * 2
        mc = ttnn.MemoryConfig(
            ttnn.TensorMemoryLayout.HEIGHT_SHARDED, ttnn.BufferType.L1,
            ttnn.ShardSpec(core_set(k, grid_x), [rows_per_core, a.cols],
                           ttnn.ShardOrientation.ROW_MAJOR))
        xd = ttnn.from_torch(torch.randn(rows, a.cols), dtype=ttnn.bfloat16,
                             layout=ttnn.TILE_LAYOUT, device=dev,
                             memory_config=ttnn.DRAM_MEMORY_CONFIG)
        s = timed(lambda: ttnn.deallocate(ttnn.to_memory_config(xd, mc)), dev)
        ttnn.deallocate(xd)
        gbs = nbytes / s / 1e9
        res["points"].append({"cores": k, "MB": round(nbytes / 1e6, 2),
                              "us": round(s * 1e6, 2), "read_GBs": round(gbs, 1),
                              "GBs_per_core": round(gbs / k, 2)})
        print(f"  cores={k:<4} {nbytes/1e6:7.2f} MB  {s*1e6:9.2f} us  {gbs:7.1f} GB/s  "
              f"{gbs/k:6.2f} GB/s per core", flush=True)

    whole = res["points"][-1]["read_GBs"]
    for p in res["points"]:
        p["frac_of_whole_grid"] = round(p["read_GBs"] / whole, 3)
        p["proportional_roof_GBs"] = round(whole * p["cores"] / res["cores_max"], 1)
    res["whole_grid_GBs"] = whole
    open(a.out, "w").write(json.dumps(res, indent=1))
    print(json.dumps(res["points"], indent=1))
    from tt_bio.tenstorrent import cleanup
    cleanup()


main()
