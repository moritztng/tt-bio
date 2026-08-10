#!/usr/bin/env python3
"""Which (grid, N) pairs make `ttnn.split_work_to_cores` throw?

`crash_scan.py` found the kernel raising `TT_FATAL work_split.cpp:305: remaining == 0` at N = 640,
960, 1280, 1600 on qb2's 11x10 grid, with the stock permute unaffected, and located it inside
`ttnn.split_work_to_cores(all_cores, Nt*Nt)` -- the wheel's own utility, called from
`reblock_permute._build`. That call is pure host-side work over a CoreRangeSet, so the grids qb2 does
not have can be tested here too. This tells the sibling leg whether qb1's 13x10 = 130 cores is
exposed, which decides whether the finding bounds the flip on the campaign host as well.

    TT_VISIBLE_DEVICES=2 python3 perf/y_permute_crossmodel/grid_scan.py
"""
from __future__ import annotations

import importlib.metadata as md
import json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import ttnn

OUT = Path(__file__).resolve().parent


def main() -> int:
    import tt_bio.tenstorrent as T
    T.get_device()  # the utility still needs the runtime initialised

    R = {"wheel": md.version("ttnn"), "grids": {}}
    for gx, gy in ((11, 10), (13, 10), (8, 8), (7, 10)):
        cores = gx * gy
        crs = ttnn.CoreRangeSet(
            [ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
        bad = []
        for nt in range(1, 65):
            groups = nt * nt
            try:
                ttnn.split_work_to_cores(crs, groups)
            except Exception:
                bad.append({"Nt": nt, "N_range": [32 * (nt - 1) + 1, 32 * nt], "groups": groups,
                            "groups_mod_cores": groups % cores})
        R["grids"][f"{gx}x{gy}"] = {"cores": cores, "failing": bad,
                                    "failing_Nt": [b["Nt"] for b in bad]}
        print(f"{gx}x{gy} ({cores} cores): failing Nt = {[b['Nt'] for b in bad]}", flush=True)
        for b in bad:
            print("   ", json.dumps(b), flush=True)
    (OUT / "grid_scan.json").write_text(json.dumps(R, indent=1))
    print("wrote", OUT / "grid_scan.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
