#!/usr/bin/env python3
"""X3 deliverable 1b: locate the untilize cliff on qb1 card 2 and test whether it is a function of
the last-dim tile count alone. Probe 1 found (9536,9536) FAST (1004 us) and 8192 cols SLOW (36026 us),
which is the opposite of what qb2 reported, so the cliff exists here at a different width."""
from __future__ import annotations
import json, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
dev = get_device()
log = lambda *a: print(*a, file=sys.stderr, flush=True)


def timed(fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        o.append(time.perf_counter() - t0)
    return st.median(o) * 1e6


def untilize(rows, cols, tag):
    try:
        t = ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    except Exception as e:
        log(f"  {tag:34s} alloc {str(e)[:50]}")
        return None
    nb = rows * cols * 2
    us = timed(lambda: ttnn.deallocate(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)))
    ttnn.deallocate(t)
    gbs = 2 * nb / (us * 1e-6) / 1e9
    log(f"  {tag:34s} {nb/2**20:6.1f} MB  {us:10.1f} us  {gbs:7.1f} GB/s")
    return {"rows": rows, "cols": cols, "MB": round(nb / 2**20, 1), "us": round(us, 1),
            "GB/s": round(gbs, 1)}


res = {}

# The shape a real fold of T tokens actually untilizes: z = (T*32, 32*T), i.e. T tile-columns.
log("=== the shape a T-token fold untilizes: (T*32, T*32) ===")
sq = {}
for T in (96, 128, 160, 192, 224, 240, 248, 254, 255, 256, 257, 258, 264, 272, 288, 298, 320, 384, 512):
    r = untilize(T * 32, T * 32, f"T={T:4d} tokens ({T} tile-cols)")
    if r:
        sq[T] = r
res["token_square"] = sq

# Is it the column count alone, or the (rows, cols) pair? Hold cols at the bad point and vary rows.
log("=== cols fixed at the bad point, rows varied ===")
rv = {}
bad = 256
for rt in (64, 128, 256, 298, 346, 512):
    r = untilize(rt * 32, bad * 32, f"rows={rt:4d} tiles, cols={bad} tiles")
    if r:
        rv[rt] = r
res["rows_at_bad_cols"] = rv

print(json.dumps(res, indent=1))
