#!/usr/bin/env python3
"""What does ONE op-boundary tile pass cost, at the Transition's own shapes?

The megakernel's whole price rests on a single conversion factor. Wave 1 measured 71.3 ns a tile
for a PACKER pass inside one kernel; a fused kernel deletes something else -- an OP BOUNDARY, which
is a pack plus a NOC move plus an unpack plus a dispatch. If the two cost the same, fusing the
Transition is worth 0.58 ms a call and the route is a tenth. If an op boundary costs 3x a packer
pass, it is worth 1.7 ms and the route is worth building out.

So price it directly, on the ops the fused kernel would delete, at the shapes it would delete them
at. Pure-movement ops only -- no matmul -- so the number is a movement cost and not a mix.

Every arm is interleaved with an A/A repeat of the first arm in the same process, so the ratio has
its own noise floor attached.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def bench(ttnn, dev, fn, reps=20):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(1e3 * (time.perf_counter() - t0))
    return st.median(ts), ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reps", type=int, default=9)
    a = ap.parse_args()
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    import tt_bio.tenstorrent as T

    dev = T.get_device()
    grid = T.COMPUTE_GRID_MAIN
    cores = grid[0] * grid[1]
    out = {"grid": list(grid), "cores": cores, "reps": a.reps,
            "ttnn": __import__("importlib.metadata", fromlist=["x"]).version("ttnn"),
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

    def mk(shape, mc):
        return ttnn.from_torch(torch.randn(*shape, dtype=torch.bfloat16), dtype=ttnn.bfloat16,
                               layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)

    L1, DR = ttnn.L1_MEMORY_CONFIG, ttnn.DRAM_MEMORY_CONFIG
    # Big enough that the ~6.4 us per-op fixed cost is under 1.5 % and the sync is amortised: the
    # first cut used the Transition's own 8.39 MB chunk and came back with a 127 % spread, which is
    # a measurement of dispatch, not of movement. Each arm is run `inner` times per timed window.
    inner = 8
    big_l1 = (1, 96, 512, 512)   # 24,576 tiles = 50.3 MB, x2 live in L1 = 100.7 MB of ~182 MB
    big_dr = (1, 512, 512, 128)  # 32,768 tiles = 67.1 MB, the pair tensor itself
    cases = []
    a1, a2 = mk(big_l1, L1), mk(big_l1, L1)
    cases.append(("multiply_ 24576t L1", lambda: [ttnn.multiply_(a1, a2) for _ in range(inner)],
                  inner * 3 * 24576))
    b1, b2 = mk(big_dr, DR), mk(big_dr, DR)
    cases.append(("multiply_ 32768t DRAM", lambda: [ttnn.multiply_(b1, b2) for _ in range(inner)],
                  inner * 3 * 32768))
    c1, c2 = mk(big_dr, DR), mk(big_dr, DR)
    cases.append(("add_ z DRAM (residual)", lambda: [ttnn.add_(c1, c2) for _ in range(inner)],
                  inner * 3 * 32768))
    e1 = mk(big_l1, L1)
    cases.append(("layer_norm 24576t L1",
                  lambda: [ttnn.deallocate(ttnn.layer_norm(e1, epsilon=1e-5, memory_config=L1))
                           for _ in range(inner)], inner * 2 * 24576))

    rows = []
    base_med = None
    for name, fn, passes in cases:
        med, ts = bench(ttnn, dev, fn, a.reps)
        if base_med is None:
            base_med = med
        ns_per_pass_per_core = 1e6 * med / passes * cores
        gbs = passes * 2048 / (med * 1e-3) / 1e9
        rows.append({"case": name, "ms": round(med, 4), "passes": passes,
                     "ns_per_pass_per_core": round(ns_per_pass_per_core, 2),
                     "spread_pct": round(100 * (max(ts) - min(ts)) / med, 2),
                     "GB_s": round(gbs, 1)})
        print("%-24s %8.4f ms  %8d passes  %8.2f ns/pass/core  %7.1f GB/s  spread %.1f %%"
              % (name, med, passes, ns_per_pass_per_core, gbs, rows[-1]["spread_pct"]), flush=True)
    # A/A floor: the first case again, at the end of the process
    med2, _ = bench(ttnn, dev, cases[0][1], a.reps)
    out["aa_floor_pct"] = round(100 * abs(med2 - base_med) / base_med, 3)
    out["rows"] = rows
    print("A/A floor %.3f %%" % out["aa_floor_pct"], flush=True)
    a.out.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
