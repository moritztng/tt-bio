"""Why the DiT's head split costs 0.196 ms to move 526 KB, and whether head_dim is the reason.

p49 put `reshape@1461` at 21.154 ms/step over 108 calls in the DiT, i.e. 0.196 ms per call on a
[1,685,384] bf16 tensor. That tensor is 526 KB, so the chain is running at ~2.7 GB/s against this
card's measured 390 GB/s read roof -- 0.7 % of it. At that ratio the cost is not bytes, it is the
kernel path, and the obvious suspect is head_dim=24: the reshape splits a 384-wide last dim into
(16, 24) and 24 is not a tile multiple, so the result cannot be a relabelling.

This is a microbenchmark, not a model change. Every arm runs the same two ops the model runs
(`reshape` then `permute(0,2,1,3)`), warm, with a sync on both sides, and the only thing that varies
is the head geometry. If the 32-wide arm is much cheaper per element, head_dim=24 is the mechanism
and the lever has a number. If every arm is equally slow, the reshape path itself is, and the fix is
a different op rather than a different shape.

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/rfd3_port/p51_head_split_micro.py
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

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "perf/p51/head_split_micro.json")
NWARM, NREP = 3, 10

# (label, length, n_head, head_dim). The first row is exactly what the DiT issues at I=685; the
# rest change one thing at a time. L=6051 is the atom-block length, where the same chain runs.
ARMS = [
    ("DiT shipped   I=685  16x24", 685, 16, 24),
    ("tile-aligned  I=685  16x32", 685, 16, 32),
    ("tile-aligned  I=685  12x32", 685, 12, 32),
    ("half-width    I=685  8x24", 685, 8, 24),
    ("atom shipped  L=6051 4x32", 6051, 4, 32),
]


def timeit(fn, dev):
    for _ in range(NWARM):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(NREP):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1e3


def main():
    dev = get_device()
    rows = []
    for label, length, n_head, head_dim in ARMS:
        c = n_head * head_dim
        x = ttnn.from_torch(torch.randn(1, length, c), dtype=ttnn.bfloat16,
                            layout=ttnn.TILE_LAYOUT, device=dev)
        mb = length * c * 2 / 1e6

        def split():
            y = ttnn.reshape(x, (1, length, n_head, head_dim))
            return ttnn.permute(y, (0, 2, 1, 3))

        def only_reshape():
            return ttnn.reshape(x, (1, length, n_head, head_dim))

        ms_both = timeit(split, dev)
        ms_reshape = timeit(only_reshape, dev)
        rows.append({"arm": label, "length": length, "n_head": n_head, "head_dim": head_dim,
                     "MB": round(mb, 3), "ms_reshape": round(ms_reshape, 4),
                     "ms_reshape_permute": round(ms_both, 4),
                     "GB_s_both": round(mb / 1e3 / (ms_both / 1e3), 1)})
        ttnn.deallocate(x)
        print("%-28s %6.3f MB  reshape %7.4f ms  +permute %7.4f ms  %7.1f GB/s"
              % (label, mb, ms_reshape, ms_both, mb / 1e3 / (ms_both / 1e3)), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "n_warm": NWARM, "n_rep": NREP,
                               "host": "qb2", "card": 0, "ttnn": "0.68.0",
                               "read_roof_GB_s_measured": 390.0}, indent=2) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
