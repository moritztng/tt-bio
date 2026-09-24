#!/usr/bin/env python3
"""Separate the SIZE effect from the TARGET effect, which this row could not do until now.

Every size number this row measured before 2026-09-24 compared one 512 crop against one 1536
crop of the same protein. Those are two different binding problems as well as two sizes, so a
single pair cannot say which axis moved. The measured across-target spread at a FIXED 512
residues turned out to be 13.04 A -- larger than the 12.34 A that had been read as a size
effect -- so the confound is not hypothetical, it dominates.

The 2x2 fixes that: {offset 0, offset 100} x {512, 1536}, all on the after-fix engine, n=8 per
cell, same designer, binder length, metric and refolder.

    size effect   = mean over offsets of (1536 cell - 512 cell)
    target effect = mean over sizes   of (offset100 cell - offset0 cell)

Both are reported with the per-cell medians they came from, and with a Mann-Whitney on the
pooled halves, because a main effect averaged over two cells is not a significance claim by
itself.

    python3 perf/mgxaccuracy/twobytwo.py perf/mgxaccuracy/results/size_1536.jsonl [...]

Reads any jsonl this row writes. A cell is identified by (target_res, crop_offset) on rows
whose `side` is the device's own scoring, so an upstream-refold row is not mixed in with a
device-refold row -- that was a separate confound and it has its own file.
"""
import argparse
import json
import pathlib
import statistics as st
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from power import mw_reject  # noqa: E402

CELLS = [(512, 0), (512, 100), (1536, 0), (1536, 100)]


def load(paths, side, target, engine):
    """Rows for the four cells, from ONE named target.

    The target filter is not optional and is not cosmetic. `size_1536.jsonl` also carries a
    1536 offset-0 row from the `big_1831` fixture -- the disconnected rung this row
    disqualified -- and it has the same (target_res, crop_offset) key as the valid one. Keying
    on size alone silently picks it up and reports 11.205 A where the valid cell reads 21.516,
    which is the same class of confound that produced a withdrawn headline on 2026-09-24:
    two measurements that look like the same cell because nothing in the key names the target.
    """
    cells, rejected = {}, []
    for p in paths:
        for line in pathlib.Path(p).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("side") != side or "scrmsd" not in d:
                continue
            key = (d.get("target_res"), d.get("crop_offset", 0))
            if key not in CELLS:
                continue
            if target not in str(d.get("target", "")):
                rejected.append((key, f"target {d.get('target')!r}"))
                continue
            # The engine era is part of a cell's identity too: the pre-fix and after-fix
            # 1536 offset-0 cells share size, offset, target, side and metric, and differ
            # only by whether cc908c377 is in the tree. A row with no `engine` key cannot be
            # placed, so it is refused rather than guessed at.
            if d.get("engine") != engine:
                rejected.append((key, f"engine {d.get('engine')!r}"))
                continue
            # A later row for the same cell supersedes an earlier one.
            cells[key] = d
    return cells, rejected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--side", default="device",
                    help="which scoring side to read (default: device)")
    ap.add_argument("--engine", default="after-fix",
                    help="required `engine` field on every row (default: after-fix). Rows "
                         "without the key are refused: pre-fix and after-fix cells are "
                         "otherwise indistinguishable by their keys")
    ap.add_argument("--target", default="gpb_dimer_1646",
                    help="substring every row's `target` must contain (default: the valid "
                         "target; rows from any other fixture are refused, not averaged in)")
    args = ap.parse_args()

    cells, rejected = load(args.jsonl, args.side, args.target, args.engine)
    print(f"side = {args.side}   target contains {args.target!r}   "
          f"engine == {args.engine!r}\n")
    for key, tgt in rejected:
        print(f"  refused {key}: {tgt}")
    if rejected:
        print()
    print("  size  offset   n   median     range")
    for key in CELLS:
        d = cells.get(key)
        if d is None:
            print(f"  {key[0]:>4}  {key[1]:>6}   -   MISSING")
            continue
        v = d["scrmsd"]
        print(f"  {key[0]:>4}  {key[1]:>6}  {len(v):>2}  {st.median(v):>7.3f}   "
              f"{min(v):.3f}-{max(v):.3f}")

    missing = [k for k in CELLS if k not in cells]
    if missing:
        print(f"\n{len(missing)} cell(s) missing: {missing}")
        print("No effect is reported until all four are in -- a main effect from half a "
              "design is the same confound this script exists to remove.")
        return 1

    med = {k: st.median(cells[k]["scrmsd"]) for k in CELLS}
    size = ((med[(1536, 0)] - med[(512, 0)]) + (med[(1536, 100)] - med[(512, 100)])) / 2
    targ = ((med[(512, 100)] - med[(512, 0)]) + (med[(1536, 100)] - med[(1536, 0)])) / 2
    inter = (med[(1536, 100)] - med[(512, 100)]) - (med[(1536, 0)] - med[(512, 0)])

    small = [x for k in CELLS if k[0] == 512 for x in cells[k]["scrmsd"]]
    big = [x for k in CELLS if k[0] == 1536 for x in cells[k]["scrmsd"]]
    off0 = [x for k in CELLS if k[1] == 0 for x in cells[k]["scrmsd"]]
    off100 = [x for k in CELLS if k[1] == 100 for x in cells[k]["scrmsd"]]

    print(f"\n  SIZE effect    {size:+8.2f} A   "
          f"(pooled n={len(small)} vs {len(big)}, MW rejects: {mw_reject(small, big)})")
    print(f"  TARGET effect  {targ:+8.2f} A   "
          f"(pooled n={len(off0)} vs {len(off100)}, MW rejects: {mw_reject(off0, off100)})")
    print(f"  interaction    {inter:+8.2f} A   "
          f"(how much the size effect itself depends on which crop)")
    print("\n  The larger of the two main effects is what dominates designability here.")
    print("  Report both with their medians; neither number means anything alone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
