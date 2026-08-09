#!/usr/bin/env python3
"""T2 probe 6 — is the channel move's cost the permute KERNEL, or the reordering itself?

Probe 5 measured a fixed 3.16 us per tile per core inside ttnn's tiled permute dataflow kernels
against 0.34 for a clone and 0.33 for `ttnn.transpose(-2,-1)` on identical tiles. `transpose(-2,-1)`
runs on the compute engine (`transpose_wh`) and tracks the clone; the permutes do not.

If the cost is the permute kernel and not the reordering, then the SAME index move expressed as
transposes should cost transpose money. The identity is exact:

    permute(0,3,1,2) == transpose(1,3) then transpose(-2,-1)
    permute(0,3,2,1) == transpose(1,3)
    permute(0,2,3,1) == transpose(-2,-1) then transpose(1,3)      [the output-side move]

All three are pure index reorderings, so every route must be `torch.equal` to the others. This
measures the cost of each route and checks that equality. It changes nothing in the model.

    PYTHONPATH=<wt> python3 perf/t2_trimul/t2_probe6.py --out <json>
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
    N, C = 320, 32
    tiles = N * N * C // 1024
    res = {"grid": f"{gx}x{gy}", "cores": ncore, "shape": [1, N, N, C], "tiles": tiles}

    x = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=L1)
    xb = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)

    def two(a, b_):
        def f():
            t = a()
            r = b_(t)
            ttnn.deallocate(t)
            return r
        return f

    routes = {
        # the production input-side move, starting variant
        "in_0312__permute": lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=L1),
        "in_0312__t13_then_twh": two(
            lambda: ttnn.transpose(x, 1, 3, memory_config=L1),
            lambda t: ttnn.transpose(t, -2, -1, memory_config=L1)),
        # the production input-side move, ending variant
        "in_0321__permute": lambda: ttnn.permute(x, (0, 3, 2, 1), memory_config=L1),
        "in_0321__t13": lambda: ttnn.transpose(x, 1, 3, memory_config=L1),
        # the production output-side move
        "out_0231__permute": lambda: ttnn.permute(xb, (0, 2, 3, 1), memory_config=L1),
        "out_0231__twh_then_t13": two(
            lambda: ttnn.transpose(xb, -2, -1, memory_config=L1),
            lambda t: ttnn.transpose(t, 1, 3, memory_config=L1)),
    }
    out = {}
    for lbl, fn in routes.items():
        try:
            s = timed(dev, fn, warm=3, pipe=4, reps=5)
            out[lbl] = {"us": round(s * 1e6, 2),
                        "us_per_tile_core": round(max(s * 1e6 - 6.40, 0.0) / (tiles / ncore), 4)}
            print(f"  {lbl:28s} {s*1e6:8.2f} us   {out[lbl]['us_per_tile_core']:6.3f} us/tile/core",
                  flush=True)
        except Exception as e:
            out[lbl] = {"err": str(e)[:110]}
            print(f"  {lbl:28s} ERR {str(e)[:110]}", flush=True)
    res["routes"] = out

    # every route must be the same tensor: pure index reordering
    print("=== bit-exactness of the routes (torch.equal) ===", flush=True)
    eq = {}
    for pair in (("in_0312__permute", "in_0312__t13_then_twh"),
                 ("in_0321__permute", "in_0321__t13"),
                 ("out_0231__permute", "out_0231__twh_then_t13")):
        try:
            a = ttnn.to_torch(routes[pair[0]]())
            b = ttnn.to_torch(routes[pair[1]]())
            eq[f"{pair[0]} vs {pair[1]}"] = {"shape_a": list(a.shape), "shape_b": list(b.shape),
                                             "torch_equal": bool(torch.equal(a, b))}
            print(f"  {pair[0]} vs {pair[1]}: shapes {list(a.shape)} / {list(b.shape)} "
                  f"torch.equal={torch.equal(a, b)}", flush=True)
        except Exception as e:
            eq[f"{pair[0]} vs {pair[1]}"] = {"err": str(e)[:110]}
            print(f"  {pair[0]} vs {pair[1]} ERR {str(e)[:110]}", flush=True)
    res["parity"] = eq

    ttnn.deallocate(x)
    ttnn.deallocate(xb)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
