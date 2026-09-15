#!/usr/bin/env python3
"""A roofs file for the part the floor of record lives on, with the shipped kernels in it.

`perf/roof_shape/shape_roofs_pc_bh.json` has two defects the 512 aa floor inherits, both measured
in `rate_ab_512_qb2c3.json`:

  * its TriangleAttention arms are `ttnn.linear` and `ttnn.transformer.scaled_dot_product_
    attention`, and the fold issues neither -- three `ttnn.generic_op` kernels do that work, and
    each one beats the stock op it was priced by (2.46x, 1.65x, 1.11x);
  * its fractions were measured on a pc p150a at a 128.70 TFLOP/s cube and carried onto a qb2
    p300c at 104.93. On the three arms that have now been measured on both parts the transfer is
    worth 1.190x on the class, against a 4 % per-core argument that said it would carry.

This writes one roofs file in the same schema so `true_floor.py --roofs` reads it unchanged.
Rows measured on qb2 card 3 in `rate_ab`'s own session are marked `native`; every other row is
carried over from the pc session at its own `pct_of_cube` and marked `transferred`, which is what
the file of record already did for all of them. The cube is the qb2 session's own, so every row's
`TFLOPs` and `pct_of_cube` stay internally consistent, and `class_rates`' `max(cand) / cube *
cube_of_record` keeps meaning what it meant.

Nothing here measures. It joins two measured files and labels which is which.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent

# rate_ab arm -> the roofs arm name it supplies. The three `_shipped` names are new and are what
# `weigh.py`'s TriangleAttention classes gained; the other three overwrite the pc rows for arms
# that have now been measured on the fold's own part.
NATIVE = {
    "ship_sdpa": "triatt_sdpa_shipped",
    "ship_in": "triatt_in_shipped",
    "ship_out": "triatt_out_shipped",
    "cat_sdpa": "triatt_sdpa_q256",
    "cat_in": "triatt_in_flat",
    "cat_out": "pair_out128_flat",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roofs", type=Path, default=PERF / "roof_shape" / "shape_roofs_pc_bh.json")
    ap.add_argument("--rates", type=Path, default=HERE / "rate_ab_512_qb2c3.json")
    ap.add_argument("--out", type=Path,
                    default=PERF / "roof_shape" / "shape_roofs_qb2c3_shipped.json")
    a = ap.parse_args()

    pc = json.loads(a.roofs.read_text())
    ab = json.loads(a.rates.read_text())
    cube = ab["arms"]["cube"]["TFLOPs_at_min"]

    rows, seen = [], set()
    for src, arm in NATIVE.items():
        r = ab["arms"][src]
        rows.append({"arm": arm, "ms": r["min_ms"], "TFLOPs": r["TFLOPs_at_min"],
                     "reps": r["reps"], "pct_of_cube": 100 * r["TFLOPs_at_min"] / cube,
                     "origin": "native", "rate_ab_arm": src})
        seen.add(arm)
    for r in pc["rows"]:
        if r["arm"] in seen:
            continue
        # carried at its own fraction, restated against this session's cube
        rows.append({"arm": r["arm"], "ms": r["ms"], "reps": r["reps"],
                     "TFLOPs": r["pct_of_cube"] / 100 * cube, "pct_of_cube": r["pct_of_cube"],
                     "origin": "transferred", "from_host": pc["host"],
                     "from_cube_TFLOPs": pc["cube4096_TFLOPs"]})
    rows.append({"arm": "cube4096", "ms": ab["arms"]["cube"]["min_ms"], "TFLOPs": cube,
                 "reps": ab["arms"]["cube"]["reps"], "pct_of_cube": 100.0, "origin": "native"})
    rows = [r for r in rows if r["arm"] != "cube4096"] + [rows[-1]]

    out = {"host": ab["meta"]["host"], "arch": ab["meta"]["arch"], "grid": ab["meta"]["grid"],
           "core_grid_used": ab["meta"]["grid"], "card": ab["meta"]["card"],
           "blocks": ab["meta"]["reps"], "loadavg": ab["meta"]["loadavg"],
           "cube4096_TFLOPs": cube, "refused": pc.get("refused", {}),
           "built_from": {"transferred": str(a.roofs.relative_to(PERF.parent)),
                          "native": str(a.rates.relative_to(PERF.parent))},
           "rows": rows}
    a.out.write_text(json.dumps(out, indent=1))

    n = sum(1 for r in rows if r["origin"] == "native")
    print("%s  cube %.2f TFLOP/s  %d native rows, %d transferred -> %s"
          % (out["host"], cube, n, len(rows) - n, a.out))
    for r in rows:
        if r["origin"] == "native":
            print("  %-22s %7.2f TFLOP/s  %5.1f %% of cube" % (r["arm"], r["TFLOPs"],
                                                               r["pct_of_cube"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
