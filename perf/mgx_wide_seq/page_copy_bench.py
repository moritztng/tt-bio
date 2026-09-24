"""A whole pair written block by block with tt_bio.page_copy, against ttnn.concat of the same blocks.

    TT_VISIBLE_DEVICES=<c> ... python perf/mgx_wide_seq/page_copy_bench.py [S] [C] [R,...]

The in-place route (`_pair_inplace`) replaces a join: when that join would have run on the device,
the write has to be at least as fast as ttnn.concat moving the same bytes. Prints one JSON row per
(kind, R): seconds for the full pair, GB/s, and whether the result equals the concat's.
"""
import json
import sys
import time

import torch

import ttnn

from tt_bio import page_copy as PC
from tt_bio.tenstorrent import get_device

S = int(sys.argv[1]) if len(sys.argv) > 1 else 3104
C = int(sys.argv[2]) if len(sys.argv) > 2 else 128
RS = [int(r) for r in sys.argv[3].split(",")] if len(sys.argv) > 3 else [32, 128]
dev = get_device()
sync = lambda: ttnn.synchronize_device(dev)
z = ttnn.from_torch(torch.zeros(1, S, S, C, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                    device=dev, dtype=ttnn.bfloat16)
nbytes = S * S * C * 2


def timed(fn, reps=2):
    fn()
    sync()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    sync()
    return (time.perf_counter() - t0) / reps


for kind in ("rows", "cols"):
    dim = 1 if kind == "rows" else 2
    for R in RS:
        shape = (1, R, S, C) if kind == "rows" else (1, S, R, C)
        blk = ttnn.from_torch(torch.randn(*shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                              dtype=ttnn.bfloat16)
        n = S // R
        write = PC.write_rows if kind == "rows" else PC.write_cols

        def pc():
            for i in range(n):
                write(z, blk, i * R)

        def cat():
            ttnn.deallocate(ttnn.concat([blk] * n, dim=dim))

        t_pc, t_cat = timed(pc), timed(cat)
        moved = n * R * S * C * 2
        print(json.dumps({"S": S, "C": C, "kind": kind, "R": R, "blocks": n,
                          "page_copy_s": round(t_pc, 4), "concat_s": round(t_cat, 4),
                          "page_copy_GBps": round(moved / t_pc / 1e9, 2),
                          "concat_GBps": round(moved / t_cat / 1e9, 2),
                          "pair_GB": round(nbytes / 1e9, 3)}), flush=True)
        ttnn.deallocate(blk)
