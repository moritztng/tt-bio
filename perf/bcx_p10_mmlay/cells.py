#!/usr/bin/env python3
"""bcx-p10-mmlay leg 1: `perf/bcx_p10_shape/shape.py` cell E with the key widened.

Cell E is the fold's own program at 288 with the `hifi` route -- the exact cell
`bcx-p10-calls` and `bcx-p10-devgap` both attributed, so this row's table and theirs are
readings of one measurement. The ONLY change is which second key `OpTimer` writes into its
`verb_*` counters: `ShapeOpTimer` widens it to shape + placement for the matmul family and
leaves every other op on its name. Nothing about the program, the cell, the reps or the
subtraction moves, which is what makes A_shape comparable to A_verb.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_p10_shape import shape as SH           # noqa: E402
from perf.bcx_p10_mmlay import mmkey as MK           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=None)
    ap.add_argument("--card", type=int, default=int(os.environ.get("TT_VISIBLE_DEVICES", "0")))
    ap.add_argument("--cells", default="E")
    ap.add_argument("--stacks", default="evo,extra")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default="cells_mmlay_E.json")
    ap.add_argument("--verb-records", action="store_true", default=True)
    args = ap.parse_args()
    if args.params is None:
        from perf.bcx_afgrad import afgrad as A
        args.params = A.DEFAULT_PARAMS
    torch.set_num_threads(args.threads)

    SH.D.OpTimer = MK.ShapeOpTimer            # the whole diff
    SH.OUT = ROOT / "perf" / "bcx_p10_mmlay" / "out"
    SH.cmd_cells(args)


if __name__ == "__main__":
    main()
