#!/usr/bin/env python3
"""The DRAM roof of one Wormhole_B0, measured the way the block's traffic is counted.

`b2z2-redteam-v2` killed the campaign's top rung by intersecting a movement-free block with a
MEASURED roof rather than a fitted asymptote, and the roof it used was `ttnn.clone` DRAM->DRAM
counted as `2N/t`: read N and write N over one interface, which is the same mix and the same
currency as a block's `dram_rd + dram_wr`. That measurement was on a Blackhole p300c processor
(390.7 GB/s). Every MSA number in this campaign is Wormhole, so the same roof has to exist on
Wormhole before any MSA headroom claim means anything.

Method is `perf/bioir_roofline/roofs_bh.py`, unchanged: a copy roof (1r+1w), a read-heavy roof
(`ttnn.add`, 2r+1w) so the two can be compared as they were on Blackhole, and a size sweep so
the answer is not one shape's accident.

    TT_VISIBLE_DEVICES=9 TT_BIO_LEASE_CARDS=9 TT_BIO_LEASE_HOLDER=worker:b2z2-msa-movement-attack \
      python3 perf/b2z2_msa_move/roof_wh.py --out perf/b2z2_msa_move/roof_wh_c9.json
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch                                                                  # noqa: E402
import ttnn                                                                   # noqa: E402

from tt_bio.tenstorrent import get_device                                     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--reps", type=int, default=7)
a = ap.parse_args()

dev = get_device()
out: dict = {"arch": str(dev.arch()), "reps": a.reps, "copy": [], "read": []}


def mk(shape):
    return ttnn.from_torch(torch.randn(*shape), dtype=ttnn.bfloat16,
                           layout=ttnn.TILE_LAYOUT, device=dev,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def bench(fn, warm=3, pipe=4):
    for _ in range(warm):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    s = []
    for _ in range(a.reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        keep = [fn() for _ in range(pipe)]
        ttnn.synchronize_device(dev)
        s.append((time.perf_counter() - t0) / pipe)
        for k in keep:
            ttnn.deallocate(k)
    return st.median(s)


# --- copy roof: 1 read + 1 write, the mix a block's dram_rd/dram_wr is counted in -------------
for shape in ([1024, 1024, 32], [1024, 1024, 64], [1024, 1024, 96], [1024, 1024, 128],
              [8192, 8192], [1024, 512, 64]):
    n = 2
    for d in shape:
        n *= d
    t = mk(shape)
    ms = bench(lambda: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3
    row = {"shape": shape, "MiB": round(n / 2**20, 1), "clone_ms": round(ms, 4),
           "copy_roof_GBps": round(2 * n / (ms * 1e-3) / 1e9, 1)}
    out["copy"].append(row)
    print("COPY " + json.dumps(row), flush=True)
    ttnn.deallocate(t)

# --- read-heavy roof: 2 reads + 1 write -------------------------------------------------------
for shape in ([1024, 1024, 128], [8192, 8192]):
    n = 2
    for d in shape:
        n *= d
    x, y = mk(shape), mk(shape)
    ms = bench(lambda: ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3
    row = {"shape": shape, "MiB": round(n / 2**20, 1), "add_ms": round(ms, 4),
           "rw_roof_GBps": round(3 * n / (ms * 1e-3) / 1e9, 1)}
    out["read"].append(row)
    print("READ " + json.dumps(row), flush=True)
    ttnn.deallocate(x)
    ttnn.deallocate(y)

out["copy_roof_GBps"] = max(r["copy_roof_GBps"] for r in out["copy"])
out["rw_roof_GBps"] = max(r["rw_roof_GBps"] for r in out["read"])
Path(a.out).write_text(json.dumps(out, indent=1))
print(f"\nCOPY ROOF {out['copy_roof_GBps']} GB/s   2r+1w ROOF {out['rw_roof_GBps']} GB/s", flush=True)
