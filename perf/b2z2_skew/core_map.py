#!/usr/bin/env python3
"""Dump the matmul operand chain: which core is the injector, and how far down each core sits.

`skew_split.py` reads device-profiler timestamps keyed by PHYSICAL core coordinates, and the hop
index it needs is a property of the LOGICAL grid walk `mm_generic._build_core_order_for_axis`
builds. Deriving the hop order from the timestamps themselves would fit the answer to the data, so
it is taken from the same code the kernels are configured by, and joined on the physical map the
device reports.

Writes {"in0": {"<px>,<py>": {"group": g, "hop": i}}, "in1": {...}} plus the grid it was taken on.
"""
import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "coremap.json")
    ap.add_argument("--transpose", action="store_true",
                    help="the transposed matmul variant; the default is the trunk's own")
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.mm_generic as MG

    dev = T.get_device()
    gx, gy = (int(v) for v in T.COMPUTE_GRID_MAIN)
    transpose = a.transpose
    in0_noc = MG.NOC_FOR_DRAM_READ if transpose else MG.NOC_FOR_DRAM_WRITE
    in1_noc = MG.NOC_FOR_DRAM_WRITE if transpose else MG.NOC_FOR_DRAM_READ
    # the length of the walk each operand's chain makes, named as mm_generic passes them:
    # in0 walks x (length gx) and in1 walks y (length gy) in the untransposed case.
    in0_walk = gy if transpose else gx
    in1_walk = gx if transpose else gy

    def phys(c):
        p = dev.worker_core_from_logical_core(ttnn.CoreCoord(c[0], c[1]))
        return int(p.x), int(p.y)

    out = {"grid": [gx, gy], "transpose": transpose,
           "card": os.environ.get("TT_VISIBLE_DEVICES"), "in0": {}, "in1": {}, "logical": {}}
    for cx in range(gx):
        for cy in range(gy):
            core = (cx, cy)
            left_core, top_core = (0, cy), (cx, 0)
            in0_order, in0_i = MG._build_core_order_for_axis(
                core, transpose, in0_walk, in0_noc, True,
                top_core if transpose else left_core)
            in1_order, in1_i = MG._build_core_order_for_axis(
                core, transpose, in1_walk, in1_noc, False,
                left_core if transpose else top_core)
            px, py = phys(core)
            key = f"{px},{py}"
            # the chain group is the set of cores sharing one injector: a row for in0, a column
            # for in1 in the untransposed case, and the other way round when transposed.
            out["in0"][key] = {"group": cx if transpose else cy, "hop": in0_i,
                               "injector": in0_i == 0, "sink": core == in0_order[-1]}
            out["in1"][key] = {"group": cy if transpose else cx, "hop": in1_i,
                               "injector": in1_i == 0, "sink": core == in1_order[-1]}
            out["logical"][key] = [cx, cy]
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"grid": [gx, gy], "cores": len(out["logical"]),
                      "in0_walk": in0_walk, "in1_walk": in1_walk,
                      "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
