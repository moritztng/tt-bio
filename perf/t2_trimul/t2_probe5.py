#!/usr/bin/env python3
"""T2 probe 5 — the channel move is on the TILED path, so what costs 6x a tile-granular clone?

Probe 4 killed the row-major-detour hypothesis: against an empty kernel cache, the production
`permute(0,3,1,2)` compiles `reader_unary_transpose_hc_interleaved_tiled_padding_aware` +
`writer_permute_interleaved_tiled_row_invariant`, and `permute(0,3,2,1)` compiles
`reader/writer_permute_interleaved_tiled_generic` + `transpose_xw_tiled`. Both are TILED. No
untilize, no tilize.

So the op moves whole 2048 B tiles and still costs 6.3x a `ttnn.clone` of the same tiles. Two
surviving explanations, and they separate on how the per-tile cost behaves with size:

  H-A  FIXED PER-TILE COST inside the kernel (address recomputation on the RISC, one unpipelined
       NOC transaction per tile). Prediction: us/tile is FLAT across a wide tile-count range and
       sits ~6x the clone's us/tile at every point.
  H-B  SCATTER / BANK CONTENTION that builds with the working set. Prediction: us/tile RISES with
       tile count, and the ratio to the clone widens.

Falsifier for both: us/tile equal to the clone's, i.e. the 6x was a fixed launch cost. Ruled out
in advance by size (both are far above the 6.40 us launch floor) but reported anyway.

    PYTHONPATH=<wt> python3 perf/t2_trimul/t2_probe5.py --out <json>
"""
import argparse
import json
import statistics as st
import time

import torch
import ttnn

from tt_bio.tenstorrent import get_device, COMPUTE_GRID_MAIN

L1 = ttnn.L1_MEMORY_CONFIG


def timed(dev, fn, warm=4, pipe=5, reps=7):
    for _ in range(warm):
        r = fn()
        del r
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(pipe)]
        ttnn.synchronize_device(dev)
        out.append((time.perf_counter() - t0) / pipe)
        del keep
    return st.median(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    gx, gy = COMPUTE_GRID_MAIN
    ncore = gx * gy
    res = {"grid": f"{gx}x{gy}", "cores": ncore}

    print("=== tile-count sweep: production channel move vs a tile-granular clone ===", flush=True)
    rows = []
    C = 32
    for L in (64, 96, 128, 160, 192, 224, 256, 320):
        tiles = L * L * C // 1024
        nb = L * L * C * 2
        rec = {"L": L, "tiles": tiles, "MB": round(nb / 1e6, 3)}
        try:
            x = ttnn.from_torch(torch.randn(1, L, L, C), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=L1)
            for lbl, fn in (
                    ("clone", lambda: ttnn.clone(x, memory_config=L1)),
                    ("transpose", lambda: ttnn.transpose(x, -2, -1, memory_config=L1)),
                    ("permute_0312", lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=L1)),
                    ("permute_0321", lambda: ttnn.permute(x, (0, 3, 2, 1), memory_config=L1)),
                    ("permute_0213", lambda: ttnn.permute(x, (0, 2, 1, 3), memory_config=L1))):
                s = timed(dev, fn, warm=3, pipe=4, reps=5)
                rec[lbl + "_us"] = round(s * 1e6, 2)
                # per-tile-per-core cost above the 6.40 us traced launch floor (T4)
                rec[lbl + "_us_per_tile_core"] = round(
                    max(s * 1e6 - 6.40, 0.0) / (tiles / ncore), 4)
                rec[lbl + "_gbps"] = round(2 * nb / s / 1e9, 1)
            ttnn.deallocate(x)
        except Exception as e:
            rec["err"] = str(e)[:90]
        rows.append(rec)
        print("  " + json.dumps(rec), flush=True)
    res["tile_sweep"] = rows

    # Does the permute's cost track the number of DISTINCT destination tiles it scatters to, or
    # just the tile count? Hold tiles fixed and change how many channels (= how many destination
    # planes) the move fans out to, at constant total bytes.
    print("=== constant bytes, varying fan-out (channels) ===", flush=True)
    fan = []
    for C2, L2 in ((8, 640), (16, 452), (32, 320), (64, 226), (128, 160)):
        tiles = L2 * L2 * C2 // 1024
        nb = L2 * L2 * C2 * 2
        rec = {"C": C2, "L": L2, "tiles": tiles, "MB": round(nb / 1e6, 3)}
        try:
            x = ttnn.from_torch(torch.randn(1, L2, L2, C2), layout=ttnn.TILE_LAYOUT, device=dev,
                                dtype=ttnn.bfloat16, memory_config=L1)
            for lbl, fn in (("clone", lambda: ttnn.clone(x, memory_config=L1)),
                            ("permute_0312",
                             lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=L1))):
                s = timed(dev, fn, warm=3, pipe=4, reps=5)
                rec[lbl + "_us"] = round(s * 1e6, 2)
                rec[lbl + "_us_per_tile_core"] = round(
                    max(s * 1e6 - 6.40, 0.0) / (tiles / ncore), 4)
            rec["ratio"] = round(rec["permute_0312_us"] / rec["clone_us"], 2)
            ttnn.deallocate(x)
        except Exception as e:
            rec["err"] = str(e)[:90]
        fan.append(rec)
        print("  " + json.dumps(rec), flush=True)
    res["fanout"] = fan

    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
