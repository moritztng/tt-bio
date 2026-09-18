#!/usr/bin/env python3
"""What row-block height does the SHIPPED `Transition` actually ask for? Source, not a census label.

Pass 18 concluded, from two rows agreeing, that the census key `1x16x512x*` is a fiction and the
fold's real pair-track chunk height is 47 rows at 512 aa (64 at 298, 32 at 768).
`c12-kblock-unlock`'s `_linear_block_cfg` witness reported `mt_total` 752/672 and "no call in the
fold has mt_total 256"; `c12-unfused-silu-bh`'s instrumented fold counted 3,762 calls where the
census books 8,960 and said `[1,16,512,128]` ships at no size.

This script evaluates the shipped expression instead of quoting either, and it disagrees with both.
The height is set at `tt_bio/tenstorrent.py:8336`:

    transition_h_chunk_size = max(1, int(_base_h * min(1.0, _ref / (w_eff * x.shape[-1]))))

with `_base_h = TRANSITION_H_CHUNK_SIZE = 16` (or `TRANSITION_H_CHUNK_SIZE_BIG = 32` when
`W <= TRANSITION_H_CHUNK_BIG_MAX_W = 384` and `c <= 256`, or `..._FAST = 32` under `_FAST_MODE`),
`_ref = 1024 * 128`, and `w_eff = W` unless `W` exceeds the W-chunking threshold, which on an
11-wide grid is `TRANSITION_W_CHUNKING_THRESHOLD = 1024`. `_IS_SMALL_GRID` is False on Blackhole.

At the fold's own configuration this gives **h_chunk = 16 and mt_total = 256 at 512 aa**, which is
EXACTLY the census label, plus 32 at 298 aa and 16 at 768 aa. So on source the census is right and
the 47 is unexplained.

This does NOT prove the two rows wrong. A measurement beats a source reading -- but only if it is a
measurement of the thing you think it is. The live possibilities, in order of likelihood: they
measured a different site (the trimul's own row-blocked projections under `PAIR_ROW_BLOCK = 128`,
or `_gp_in_chunks`) rather than `Transition`; or a different configuration (`_FAST_MODE`, a
`TT_BIO_TRANSITION_*` env override, a different grid); or their witness reported a quantity that is
not chunk rows. `c12-profiled-fold` settles it by printing the issued shape in situ.

Constants are read from the checked-out source rather than hardcoded, so this cannot silently drift.
"""
import re
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "tt_bio" / "tenstorrent.py"


def consts(text):
    out = {}
    for name in ("TRANSITION_H_CHUNK_SIZE", "TRANSITION_H_CHUNK_SIZE_FAST",
                 "TRANSITION_H_CHUNK_SIZE_BIG", "TRANSITION_H_CHUNK_BIG_MAX_W",
                 "TRANSITION_W_CHUNKING_THRESHOLD", "TRANSITION_W_CHUNK_SIZE",
                 "SEQ_LEN_MORE_CHUNKING", "COMPUTE_GRID_X_11", "COMPUTE_GRID_X_13"):
        m = re.search(rf"^{name}\s*=\s*(\d+)", text, re.M)
        if not m:
            sys.exit(f"constant {name} not found in {SRC} -- source moved, fix this script")
        out[name] = int(m.group(1))
    return out


def chunk_height(W, c, k, fast=False, grid_x=11):
    """The shipped expression, with the same branch order as `Transition.__call__`."""
    base = k["TRANSITION_H_CHUNK_SIZE_FAST"] if fast else k["TRANSITION_H_CHUNK_SIZE"]
    if not fast and W <= k["TRANSITION_H_CHUNK_BIG_MAX_W"] and c <= 256:
        base = k["TRANSITION_H_CHUNK_SIZE_BIG"]
    ref = 1024 * 128                                    # _IS_SMALL_GRID is False on Blackhole
    thr = (k["SEQ_LEN_MORE_CHUNKING"] if grid_x == k["COMPUTE_GRID_X_13"]
           else k["TRANSITION_W_CHUNKING_THRESHOLD"])
    w_eff = min(W, k["TRANSITION_W_CHUNK_SIZE"]) if W > thr else W
    return base, w_eff, max(1, int(base * min(1.0, ref / (w_eff * c))))


def main():
    k = consts(SRC.read_text())
    print(f"source: {SRC}")
    print("  " + "  ".join(f"{n}={v}" for n, v in k.items() if n.startswith("TRANSITION_H")))
    # (size, pair width W, channel c) -- W is the padded token count of the pair track
    rows = [(298, 320, 128), (512, 512, 128), (768, 768, 128)]
    claimed = {298: 64, 512: 47, 768: 32}               # pass-18 figure, from the two rows
    print(f"\n{'aa':>5}{'W':>6}{'c':>5}{'base':>6}{'w_eff':>7}"
          f"{'h_chunk':>9}{'mt_total':>10}{'rows claim':>12}{'agree':>7}")
    bad = 0
    for aa, W, c in rows:
        base, w_eff, h = chunk_height(W, c, k)
        mt = h * W // 32
        ok = h == claimed[aa]
        bad += not ok
        print(f"{aa:5d}{W:6d}{c:5d}{base:6d}{w_eff:7d}{h:9d}{mt:10d}"
              f"{claimed[aa]:12d}{'yes' if ok else 'NO':>7}")
    print(f"\n{bad} of {len(rows)} sizes disagree with the figure pass 18 adopted.")
    print("At 512 aa the shipped expression gives h_chunk=16, mt_total=256 -- exactly the census")
    print("label `1x16x512x*`. c12-kblock-unlock's witness reported mt_total 752/672 and said no")
    print("call in the fold has 256. Both cannot be describing the same call site.")
    print("SETTLE IT IN SITU (c12-profiled-fold), do not settle it by argument.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
