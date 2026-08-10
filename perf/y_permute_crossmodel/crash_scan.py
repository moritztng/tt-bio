#!/usr/bin/env python3
"""N=1280 threw inside the ladder. Isolate it: kernel-only vs stock-only, one call each, over the
whole eligible DRAM range on this 11x10 grid.

The shipped gate says `DRAM and N >= 256` is eligible. If the kernel THROWS at some N in that range
on a 110-core grid, the flip is not merely a perf question there -- it is a crash, and the gate needs
a bound. This decides whether that is the kernel's `ttnn.split_work_to_cores(all_cores, Nt*Nt)` or
the wheel's own permute.

    TT_VISIBLE_DEVICES=2 python3 perf/y_permute_crossmodel/crash_scan.py
"""
from __future__ import annotations

import importlib.metadata as md
import json, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

OUT = Path(__file__).resolve().parent
C = 32


def main() -> int:
    import tt_bio.tenstorrent as T
    from tt_bio import reblock_permute as RP

    dev = T.get_device()
    g = dev.compute_with_storage_grid_size()
    cores = g.x * g.y
    RP.set_enabled(True)
    R = {"wheel": md.version("ttnn"), "host": "qb2", "grid": [g.x, g.y], "cores": cores, "C": C,
         "rows": []}
    mc = ttnn.DRAM_MEMORY_CONFIG

    # split_work_to_cores(110 cores, G groups) is the suspect. Scan every tile count Nt from 8 to 56
    # (N = 32*Nt), which covers every DRAM-eligible pair size a tt-bio fold can reach.
    for nt in range(8, 57):
        N = 32 * nt
        groups = nt * nt
        row = {"N": N, "Nt": nt, "groups": groups}
        # 1. does the pure work split throw, with no tensors at all?
        try:
            all_cores = ttnn.CoreRangeSet(
                [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(g.x - 1, g.y - 1))])
            ttnn.split_work_to_cores(all_cores, groups)
            row["split_work_to_cores"] = "ok"
        except Exception as e:
            row["split_work_to_cores"] = type(e).__name__ + ": " + str(e).split("\n")[0][:90]
        # 2. the kernel and the stock call, one call each
        try:
            x = ttnn.from_torch(torch.zeros(1, N, N, C, dtype=torch.bfloat16),
                                layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
        except Exception as e:
            row["alloc"] = type(e).__name__
            R["rows"].append(row)
            print(json.dumps(row), flush=True)
            continue
        for name, fn in (("kernel", lambda: RP.reblock_permute(x, mc)),
                         ("stock", lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=mc))):
            try:
                y = fn()
                ttnn.synchronize_device(dev)
                ttnn.deallocate(y)
                row[name] = "ok"
            except Exception as e:
                row[name] = type(e).__name__ + ": " + str(e).split("\n")[0][:90]
        ttnn.deallocate(x)
        R["rows"].append(row)
        print(json.dumps(row), flush=True)
        (OUT / "crash_scan.json").write_text(json.dumps(R, indent=1))

    bad = [r for r in R["rows"] if r.get("kernel", "ok") != "ok"]
    R["kernel_failures"] = bad
    R["kernel_failure_Ns"] = [r["N"] for r in bad]
    print("KERNEL FAILURES:", json.dumps(R["kernel_failure_Ns"]), flush=True)
    (OUT / "crash_scan.json").write_text(json.dumps(R, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
