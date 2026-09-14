#!/usr/bin/env python3
"""Does the above-cap route serve, at the seven lengths row A3 names, on each architecture?

Host only. `ladder` and `best_fused` are imported from `perf/bgsdpa/fused_reach.py`, so this asks
the shipped `fused_pairs` / `q_parallel_factor` rather than re-deriving them. The only thing that
differs between a Blackhole and a Wormhole part here is the core count: `sdpa_generic.plan` reads
the grid as a core count, the CB tile counts carry no grid term, and the per-core L1 is 1.5 MiB on
both. So the sweep is over (heads, cores), and the part names are labels on core counts.

    python3 perf/ttx_a3/reach7.py --out perf/ttx_a3/reach_7lengths.json
"""
import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf" / "bgsdpa"))
_spec = importlib.util.spec_from_file_location(
    "_fused_reach", REPO / "perf" / "bgsdpa" / "fused_reach.py")
FR = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FR)

WANT = [1024, 1280, 1536, 1792, 2048, 2304, 2592]
PARTS = [(130, "Blackhole p150a, 13x10 custom fw"),
         (110, "Blackhole p300c / p150a, 11x10"),
         (72, "Wormhole Galaxy, 8x9"),
         (64, "Wormhole n150/n300, 8x8")]
SHIPPED_CAP = 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    res = {}
    for heads in (4, 8, 12):
        for cores, label in PARTS:
            rows = {}
            for S in WANT:
                shipped, _ = FR.ladder(S, heads, cores, SHIPPED_CAP)
                raised, _ = FR.ladder(S, heads, cores, 1 << 20)
                route = FR.best_fused(S, heads, cores, 0)
                rows[S] = {"shipped_ladder": list(shipped),
                           "cap_raised_ladder": list(raised),
                           "this_route": route}
            res[f"h{heads}_c{cores}"] = {"cores": cores, "heads": heads, "part": label,
                                         "lengths": rows}
    a.out.write_text(json.dumps(res, indent=1) + "\n")

    print(f"{'heads':>5} {'cores':>5} " + " ".join(f"{S:>10}" for S in WANT))
    for key, v in res.items():
        cells = []
        for S in WANT:
            r = v["lengths"][S]["this_route"]
            cells.append("-" if not r else f"q{r[0]}k{r[1]}")
        print(f"{v['heads']:>5} {v['cores']:>5} " + " ".join(f"{c:>10}" for c in cells))
    print("\ncells are the q_chunk and k_chunk this route serves; '-' is no fused pair, so the "
          "stock ladder keeps the call")
    print("\nL1 bytes of the pair it picks, against the 1572864 B per-core budget:")
    for key, v in res.items():
        b = [(v["lengths"][S]["this_route"] or [None, None, None])[2] for S in WANT]
        print(f"{v['heads']:>5} {v['cores']:>5} " + " ".join(
            f"{('-' if x is None else x):>10}" for x in b))
    return 0


sys.exit(main())
