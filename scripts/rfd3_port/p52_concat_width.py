"""Is the 258-wide concat slow because 258 is not a tile multiple, and what is its floor?

p46 timed `ttnn.concat[B,I,I,258]` at 25.35 ms/call, 50.7 ms/step, 22.5 GB/s -- 5.8 % of this card's
measured 390 GB/s read roof, the worst efficiency of any op in the model. E3.1 already killed
DELETING it (the fold-the-gathers lever, refuted by 2.2x). This asks a different question that the
kill does not cover: whether the 258 width itself is the cost, and what any implementation would
have to pay just to materialise the output.

The concat is z (128) with two one-hots (65 and 65). None of 65, 130 or 258 is a multiple of 32.

  A  128 + 65 + 65  -> 258   what ships
  B  128 + 96 + 96  -> 320   every part tile-aligned, 24 % MORE bytes
  C  128 + 64 + 64  -> 256   tile-aligned and slightly FEWER bytes; not equivalent to A, a probe
  D  clone of the 258 output  the floor: what it costs merely to write the result

Microbenchmark at the production shape, warm, synced both sides. No model change.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p52_concat_width.py
"""
import json
import os
import pathlib
import statistics
import sys
import time

import torch
import ttnn

sys.path.insert(0, os.getcwd())
from tt_bio.tenstorrent import get_device                              # noqa: E402

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p52/concat_width.json")
I = 685
NWARM, NREP = 2, 6
ARMS = [("A shipped   128+65+65", [128, 65, 65]),
        ("B aligned   128+96+96", [128, 96, 96]),
        ("C aligned   128+64+64", [128, 64, 64])]


def timeit(fn, dev):
    for _ in range(NWARM):
        t = fn()
        ttnn.deallocate(t)
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(NREP):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        t = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(t)
    return statistics.median(ts) * 1e3


def main():
    dev = get_device()
    rows = []
    for label, widths in ARMS:
        parts = [ttnn.from_torch(torch.randn(1, I, I, w), dtype=ttnn.bfloat16,
                                 layout=ttnn.TILE_LAYOUT, device=dev) for w in widths]
        total_w = sum(widths)
        mb_out = I * I * total_w * 2 / 1e6
        ms = timeit(lambda: ttnn.concat(parts, dim=-1), dev)
        rows.append({"arm": label, "widths": widths, "out_width": total_w,
                     "MB_out": round(mb_out, 1), "ms": round(ms, 4),
                     "GB_s_rw": round(2 * mb_out / 1e3 / (ms / 1e3), 1)})
        print("%-24s out %3d  %6.1f MB  %8.4f ms  %6.1f GB/s read+write"
              % (label, total_w, mb_out, ms, 2 * mb_out / 1e3 / (ms / 1e3)), flush=True)
        for t in parts:
            ttnn.deallocate(t)

    ref = ttnn.from_torch(torch.randn(1, I, I, 258), dtype=ttnn.bfloat16,
                          layout=ttnn.TILE_LAYOUT, device=dev)
    mb_out = I * I * 258 * 2 / 1e6
    ms = timeit(lambda: ttnn.clone(ref), dev)
    rows.append({"arm": "D clone of the 258 output", "widths": [258], "out_width": 258,
                 "MB_out": round(mb_out, 1), "ms": round(ms, 4),
                 "GB_s_rw": round(2 * mb_out / 1e3 / (ms / 1e3), 1)})
    print("%-24s out %3d  %6.1f MB  %8.4f ms  %6.1f GB/s read+write"
          % ("D clone of 258", 258, mb_out, ms, 2 * mb_out / 1e3 / (ms / 1e3)), flush=True)
    ttnn.deallocate(ref)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "tokens": I, "n_warm": NWARM, "n_rep": NREP,
                               "host": "qb2", "card": 0, "ttnn": "0.68.0",
                               "read_roof_GB_s_measured": 390.0,
                               "write_roof_GB_s_measured": 269.6}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
