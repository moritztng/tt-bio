#!/usr/bin/env python3
"""process_z collapse gather microbench: the shipped row-major embedding + to_layout vs a tiled embedding.

    gap_pz_embed.py OUT.json [I=685] [N=65] [REPS=30]

A (shipped): idx [1, I*I] -> embedding ROW_MAJOR -> reshape [1,I,I,C] -> to_layout TILE.
B: idx [I, I] -> embedding TILE_LAYOUT -> reshape [1,I,I,C] (leading dim only, no retile).
Both are pure gathers, so B must equal A bit for bit (torch.equal); ms/call synchronized.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())
from tt_bio.main import ensure_p300_mesh_descriptor                       # noqa: E402

ensure_p300_mesh_descriptor()
import torch                                                              # noqa: E402
import ttnn                                                               # noqa: E402
from tt_bio import tenstorrent as T                                       # noqa: E402

OUT = sys.argv[1]
I = int(sys.argv[2]) if len(sys.argv) > 2 else 685
N = int(sys.argv[3]) if len(sys.argv) > 3 else 65
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
C = 128
torch.manual_seed(0)
dev = T.get_device()
tab = ttnn.from_torch(torch.randn(N, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
bins = torch.randint(0, N, (I, I), dtype=torch.int32)
idx_a = ttnn.from_torch(bins.reshape(1, -1), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
idx_b = ttnn.from_torch(bins, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)


def arm_a():
    em = ttnn.embedding(idx_a, tab, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.to_layout(ttnn.reshape(em, (1, I, I, C)), ttnn.TILE_LAYOUT)


def arm_b():
    em = ttnn.embedding(idx_b, tab, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(em, (1, I, I, C))


res = {"I": I, "N": N, "reps": REPS, "card": os.environ.get("TT_VISIBLE_DEVICES")}
outs = {}
for name, fn in (("A_shipped", arm_a), ("B_tiled", arm_b)):
    outs[name] = ttnn.to_torch(fn())
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        fn()
    ttnn.synchronize_device(dev)
    res[name] = {"ms_per_call": 1e3 * (time.perf_counter() - t0) / REPS}
    print(name, res[name], flush=True)
res["equal"] = bool(torch.equal(outs["A_shipped"], outs["B_tiled"]))
print("equal", res["equal"])
json.dump(res, open(OUT, "w"), indent=1)
