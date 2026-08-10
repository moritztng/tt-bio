#!/usr/bin/env python3
"""Addendum to window_ladder.py — the DRAM shape class an opendde-abag fold runs.

opendde-abag was not folded this pass (three MSA-dependent chains, no offline a3m on this box). Its
refiner runs the pair track at ~1.95x the residue count, so the release gate's 891-residue leg lands
near N=1740 on the DRAM path. This extends the ladder over that range so the abag verdict rests on a
measurement of its shape class rather than on an inference from the code path. DRAM only; C=32.

    TT_VISIBLE_DEVICES=2 python3 perf/y_permute_crossmodel/ladder_abag.py
"""
from __future__ import annotations

import importlib.metadata as md
import json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

OUT = Path(__file__).resolve().parent
K = 8
C = 32


def main() -> int:
    import tt_bio.tenstorrent as T
    from tt_bio import reblock_permute as RP

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    cores = g.x * g.y
    RP.set_enabled(True)
    R = {"wheel": md.version("ttnn"), "host": "qb2", "card": os.environ.get("TT_VISIBLE_DEVICES"),
         "grid": [g.x, g.y], "cores": cores, "C": C, "buffer": "DRAM", "rows": []}

    def timed(fn, x, mc):
        for _ in range(2):
            ttnn.deallocate(fn(x, mc))
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        ys = [fn(x, mc) for _ in range(K)]
        ttnn.synchronize_device(dev)
        thru = (time.perf_counter() - t0) * 1e6 / K
        for y in ys:
            ttnn.deallocate(y)
        ss = []
        for _ in range(3):
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
    mc = ttnn.DRAM_MEMORY_CONFIG

    for N in (1024, 1280, 1536, 1740):
        nt = (N + 31) // 32
        try:
            x = ttnn.from_torch(torch.randn(1, N, N, C, dtype=torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
            kt, ks = timed(kern, x, mc)
            st, ss = timed(stock, x, mc)
            yk = RP.reblock_permute(x, mc)
            yp = ttnn.permute(x, (0, 3, 1, 2), memory_config=mc)
            eq = bool(torch.equal(ttnn.to_torch(yk), ttnn.to_torch(yp)))
            ttnn.deallocate(yk); ttnn.deallocate(yp); ttnn.deallocate(x)
            row = {"N": N, "groups": nt * nt, "groups_per_core": round(nt * nt / cores, 2),
                   "MB_one_way": round(N * N * C * 2 / 1e6, 1),
                   "kernel_thru_us": round(kt, 2), "stock_thru_us": round(st, 2),
                   "thru_ratio_stock_over_kernel": round(st / kt, 4),
                   "kernel_synced_us": round(ks, 2), "stock_synced_us": round(ss, 2),
                   "synced_ratio": round(ss / ks, 4), "torch_equal": eq}
        except Exception as e:
            row = {"N": N, "groups": nt * nt, "error": type(e).__name__, "msg": str(e)[:120]}
        R["rows"].append(row)
        print(json.dumps(row), flush=True)
        (OUT / "ladder_abag.json").write_text(json.dumps(R, indent=1))
    print("wrote", OUT / "ladder_abag.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
