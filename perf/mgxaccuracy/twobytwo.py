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

def load(paths, side, target, engine, cells_wanted):
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
            if key not in cells_wanted:
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
    # The two crops are a property of the TARGET, not of the design: 1GPB's 1646 residues
    # admit 0 and 100, GroEL's 2096 admit 0 and 560, and a hardcoded 100 finds no GroEL cell
    # at all rather than failing loudly.
    ap.add_argument("--offsets", default="0,100",
                    help="the two crop offsets, low first (default 0,100 for gpb_dimer_1646; "
                         "GroEL's most separated legal 1536 pair is 0,560)")
    ap.add_argument("--sizes", default="512,1536",
                    help="the two target sizes, small first (default 512,1536)")
    args = ap.parse_args()

    off0, off1 = (int(x) for x in args.offsets.split(","))
    lo, hi = (int(x) for x in args.sizes.split(","))
    CELLS = [(lo, off0), (lo, off1), (hi, off0), (hi, off1)]

    cells, rejected = load(args.jsonl, args.side, args.target, args.engine, CELLS)
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
    size = ((med[(hi, off0)] - med[(lo, off0)]) + (med[(hi, off1)] - med[(lo, off1)])) / 2
    targ = ((med[(lo, off1)] - med[(lo, off0)]) + (med[(hi, off1)] - med[(hi, off0)])) / 2
    inter = (med[(hi, off1)] - med[(lo, off1)]) - (med[(hi, off0)] - med[(lo, off0)])

    small = [x for k in CELLS if k[0] == lo for x in cells[k]["scrmsd"]]
    big = [x for k in CELLS if k[0] == hi for x in cells[k]["scrmsd"]]
    a_off0 = [x for k in CELLS if k[1] == off0 for x in cells[k]["scrmsd"]]
    a_off1 = [x for k in CELLS if k[1] == off1 for x in cells[k]["scrmsd"]]

    # SIMPLE effects first. Each is a clean n=8 vs n=8 comparison holding the other factor
    # fixed, which is the only test here that a Mann-Whitney is straightforwardly valid for.
    print("\n  SIMPLE effects (the other factor held fixed, n=8 vs n=8):")
    for off in (off0, off1):
        a, b = cells[(lo, off)]["scrmsd"], cells[(hi, off)]["scrmsd"]
        print(f"    size   at offset {off:>3}   {med[(hi, off)] - med[(lo, off)]:+8.2f} A"
              f"   MW rejects: {mw_reject(a, b)}")
    for size_ in (lo, hi):
        a, b = cells[(size_, off0)]["scrmsd"], cells[(size_, off1)]["scrmsd"]
        print(f"    crop   at {size_:>4}        "
              f"{med[(size_, off1)] - med[(size_, off0)]:+8.2f} A   MW rejects: {mw_reject(a, b)}")

    print(f"\n  MAIN effects (mean of the two simple effects):")
    print(f"    SIZE           {size:+8.2f} A")
    print(f"    TARGET         {targ:+8.2f} A")
    print(f"    interaction    {inter:+8.2f} A   "
          f"(how much the size effect depends on which crop)")

    # The pooled tests are printed last and deliberately hedged. Pooling two cells that differ
    # by more than the effect under test makes each pooled group bimodal, and a Mann-Whitney on
    # a bimodal group answers a question nobody asked. Kept because it is what an earlier
    # version of this row would have quoted, and it should be visible next to the honest test.
    print(f"\n  pooled, FOR REFERENCE ONLY -- each group is bimodal when the interaction is")
    print(f"  large, so read the simple effects above instead:")
    print(f"    size   pooled n={len(small)} vs {len(big)}   MW rejects: {mw_reject(small, big)}")
    print(f"    crop   pooled n={len(a_off0)} vs {len(a_off1)}   MW rejects: "
          f"{mw_reject(a_off0, a_off1)}")

    print("\n  Report the larger main effect as what dominates designability, with the two")
    print("  simple effects behind it. Neither main effect means anything alone, and the")
    print("  size axis here also changes chain count (see results/across_target_floor_512.txt).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
