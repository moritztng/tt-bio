#!/usr/bin/env python3
"""Dump the matmul operand chain: which core is the injector, and how far down each core sits.

The hop index `skew_split.py` needs is a property of the LOGICAL grid walk
`mm_generic._build_core_order_for_axis` builds, so it is taken from the same code the kernels are
configured by rather than inferred from the timestamps it is supposed to explain.

Written keyed by LOGICAL core, because that is the space the walk is defined in. The device
profiler log reports physical coordinates and `ttnn` exposes no physical mapping, so `skew_split.py`
joins by rank -- translated coordinates enumerate the unharvested workers in physical order, so the
i-th distinct physical column is logical column i -- and checks that join against which cores
actually issued a DRAM read.

Writes {"in0": {"<lx>,<ly>": {"group": g, "hop": i}}, "in1": {...}} plus the grid it was taken on.
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
    a = ap.parse_args()

    import ttnn
    import tt_bio.tenstorrent as T
    import tt_bio.mm_generic as MG

    dev = T.get_device()
    gx, gy = (int(v) for v in T.COMPUTE_GRID_MAIN)

    def phys(c):
        p = dev.worker_core_from_logical_core(ttnn.CoreCoord(c[0], c[1]))
        return int(p.x), int(p.y)

    # `build` picks the geometry per matmul with `transpose = M > N`, and both appear inside one
    # Pairformer block, so both walks are written out and `skew_split.py` picks the one whose
    # injector set matches the cores that actually read DRAM in that program.
    out = {"grid": [gx, gy], "card": os.environ.get("TT_VISIBLE_DEVICES"), "variants": {}}
    for transpose in (False, True):
        in0_noc = MG.NOC_FOR_DRAM_READ if transpose else MG.NOC_FOR_DRAM_WRITE
        in1_noc = MG.NOC_FOR_DRAM_WRITE if transpose else MG.NOC_FOR_DRAM_READ
        in0_walk = gy if transpose else gx
        in1_walk = gx if transpose else gy
        var = {"in0": {}, "in1": {}, "in0_walk": in0_walk, "in1_walk": in1_walk}
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
                key = f"{cx},{cy}"
                # the chain group is the set of cores sharing one injector: a grid row for a walk
                # along x, a column for a walk along y.
                var["in0"][key] = {"group": cx if transpose else cy, "hop": in0_i,
                                   "injector": in0_i == 0, "sink": core == in0_order[-1]}
                var["in1"][key] = {"group": cy if transpose else cx, "hop": in1_i,
                                   "injector": in1_i == 0, "sink": core == in1_order[-1]}
        out["variants"]["transpose" if transpose else "plain"] = var
    out["virtual"] = {f"{cx},{cy}": list(phys((cx, cy)))
                      for cx in range(gx) for cy in range(gy)}
    a.out.write_text(json.dumps(out, indent=1))
    print(json.dumps({"grid": [gx, gy], "cores": len(out["virtual"]),
                      "variants": {k: [v["in0_walk"], v["in1_walk"]]
                                   for k, v in out["variants"].items()},
                      "out": str(a.out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
