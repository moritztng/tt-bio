#!/usr/bin/env python3
"""Deliverable 3b — where the gate's window edges sit on qb2's 11x10 = 110-core grid.

The shipped window (`(DRAM and N>=256) or (L1 and 288<=N<=352)`) was measured on qb1's 13x10 = 130
cores. Its edges come from group-count arithmetic: the kernel splits `ceil(N/32)^2` row-groups over
the grid, so the cost steps whenever the group count crosses a multiple of the core count. On 110
cores those crossings are at different N than on 130:

    groups >= cores          : ceil(N/32) >= 11  ->  N >= 321   (130 cores: N >= 353)
    groups >= 2 x cores      : ceil(N/32) >= 15  ->  N >= 449   (130 cores: N >= 513)

This measures it instead of arguing it. `thru` enqueues K calls and syncs once, which is what a fold
does; `synced` syncs around each call. Both reported. Everything is a qb2 / 0.68.0 ratio.

    TT_VISIBLE_DEVICES=2 python3 perf/y_permute_crossmodel/window_ladder.py
"""
from __future__ import annotations

import importlib.metadata as md
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

OUT = Path(__file__).resolve().parent
K = 12
C = 32


def main() -> int:
    import tt_bio.tenstorrent as T
    from tt_bio import reblock_permute as RP

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    cores = g.x * g.y
    RP.set_enabled(True)

    R = {"wheel": md.version("ttnn"), "host": "qb2", "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "cores": cores, "C": C, "reps_per_timed_region": K, "rows": []}

    def timed(fn, x, mc):
        # warm
        for _ in range(2):
            ttnn.deallocate(fn(x, mc))
        ttnn.synchronize_device(dev)
        # thru: enqueue K, sync once
        t0 = time.perf_counter()
        ys = [fn(x, mc) for _ in range(K)]
        ttnn.synchronize_device(dev)
        thru = (time.perf_counter() - t0) * 1e6 / K
        for y in ys:
            ttnn.deallocate(y)
        # synced: sync around each
        ss = []
        for _ in range(5):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            y = fn(x, mc)
            ttnn.synchronize_device(dev)
            ss.append((time.perf_counter() - t0) * 1e6)
            ttnn.deallocate(y)
        ss.sort()
        return thru, ss[len(ss) // 2]

    kern = lambda x, mc: RP.reblock_permute(x, mc)
    stock = lambda x, mc: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)

    for N in (256, 288, 298, 320, 352, 384, 416, 448, 512, 576, 640, 704, 896):
        nt = (N + 31) // 32
        groups = nt * nt
        for bt, mc in (("L1", ttnn.L1_MEMORY_CONFIG), ("DRAM", ttnn.DRAM_MEMORY_CONFIG)):
            mb = N * N * C * 2 / 1e6
            if bt == "L1" and mb > 45:
                R["rows"].append({"N": N, "buffer": bt, "groups": groups, "MB_one_way": round(mb, 2),
                                  "skipped": "L1 output too large for this probe"})
                continue
            try:
                x = ttnn.from_torch(torch.randn(1, N, N, C, dtype=torch.bfloat16),
                                    layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
                kt, ks = timed(kern, x, mc)
                st, ss = timed(stock, x, mc)
                # bit-exactness at this shape, both arms
                yk = RP.reblock_permute(x, mc)
                yp = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
                eq = bool(torch.equal(ttnn.to_torch(yk), ttnn.to_torch(yp)))
                ttnn.deallocate(yk); ttnn.deallocate(yp); ttnn.deallocate(x)
                row = {"N": N, "buffer": bt, "groups": groups,
                       "groups_per_core": round(groups / cores, 2), "MB_one_way": round(mb, 2),
                       "kernel_thru_us": round(kt, 2), "stock_thru_us": round(st, 2),
                       "thru_ratio_stock_over_kernel": round(st / kt, 4),
                       "kernel_synced_us": round(ks, 2), "stock_synced_us": round(ss, 2),
                       "synced_ratio": round(ss / ks, 4),
                       "in_shipped_window": bool((bt == "DRAM" and N >= 256)
                                                 or (bt == "L1" and 288 <= N <= 352)),
                       "torch_equal": eq}
            except Exception as e:
                row = {"N": N, "buffer": bt, "groups": groups, "error": type(e).__name__}
            R["rows"].append(row)
            print(json.dumps(row), flush=True)
            (OUT / "window_ladder.json").write_text(json.dumps(R, indent=1))
    print("wrote", OUT / "window_ladder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
