#!/usr/bin/env python3
"""process_z collapse gather microbench: the shipped row-major embedding + to_layout vs a tiled embedding.

    gap_pz_embed.py OUT.json [I=685] [N=65] [REPS=30] [IDX=rand|rel]

A (shipped): idx [1, I*I] -> embedding ROW_MAJOR -> reshape [1,I,I,C] -> to_layout TILE.
B: idx [I, I] -> embedding TILE_LAYOUT -> reshape [1,I,I,C] (leading dim only, no retile).
A_l1 / B_l1: the same with the table held in L1 instead of DRAM (the gather reads one 256-byte row per pair, so
it is bound by DRAM read latency, not bytes: WH 5 ms/call against a 0.5 ms roof in cen2).
G: tt_bio.gather_tiled (row reads in flight per 32-row block, tilized on the unpacker, TILE out in one program).
S: G on a gather_tiled.spread table (12 replicas, one per DRAM bank).
IDX=rel replaces the random index with RFD3's clipped relative-position bins (N=65: clamp(j-i+32, 0, 64); N=4225:
that bin paired with itself), which name the two end rows for most pairs: the in-model pattern.
All are pure gathers, so every arm must equal A bit for bit (torch.equal); ms/call synchronized.
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
from tt_bio import gather_tiled as G                                      # noqa: E402

OUT = sys.argv[1]
I = int(sys.argv[2]) if len(sys.argv) > 2 else 685
N = int(sys.argv[3]) if len(sys.argv) > 3 else 65
REPS = int(sys.argv[4]) if len(sys.argv) > 4 else 30
IDX = sys.argv[5] if len(sys.argv) > 5 else "rand"
C = 128
torch.manual_seed(0)
dev = T.get_device()
tab = ttnn.from_torch(torch.randn(N, C), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
if IDX == "rel":
    ar = torch.arange(I)
    rel = (ar[None, :] - ar[:, None] + 32).clamp(0, 64).to(torch.int32)
    assert N in (65, 4225), N
    bins = rel if N == 65 else rel * 65 + rel
else:
    bins = torch.randint(0, N, (I, I), dtype=torch.int32)
idx_a = ttnn.from_torch(bins.reshape(1, -1), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
idx_b = ttnn.from_torch(bins, layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)
tab_l1 = ttnn.to_memory_config(tab, ttnn.L1_MEMORY_CONFIG)
tab_rm = ttnn.to_layout(tab, ttnn.ROW_MAJOR_LAYOUT)
tab_sp = G.spread(tab_rm)
idx_g = ttnn.from_torch(G.pad_index(bins, I, I), layout=ttnn.ROW_MAJOR_LAYOUT, device=dev, dtype=ttnn.uint32)


def arm_a(t=tab):
    em = ttnn.embedding(idx_a, t, layout=ttnn.ROW_MAJOR_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.to_layout(ttnn.reshape(em, (1, I, I, C)), ttnn.TILE_LAYOUT)
    ttnn.deallocate(em)
    return out


def arm_b(t=tab):
    em = ttnn.embedding(idx_b, t, layout=ttnn.TILE_LAYOUT, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    return ttnn.reshape(em, (1, I, I, C))


res = {"I": I, "N": N, "reps": REPS, "idx": IDX, "card": os.environ.get("TT_VISIBLE_DEVICES")}
outs = {}
for name, fn in (("A_shipped", arm_a), ("B_tiled", arm_b),
                 ("A_l1", lambda: arm_a(tab_l1)), ("B_l1", lambda: arm_b(tab_l1)),
                 ("G_gather_tiled", lambda: G.gather_tiled(idx_g, tab_rm, 1, I)),
                 ("S_spread", lambda: G.gather_tiled(idx_g, tab_sp, 1, I, copies=G.SPREAD_COPIES))):
    outs[name] = ttnn.to_torch(fn())
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    for _ in range(REPS):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    res[name] = {"ms_per_call": 1e3 * (time.perf_counter() - t0) / REPS,
                 "equal_A": bool(torch.equal(outs["A_shipped"], outs[name]))}
    print(name, res[name], flush=True)
res["equal"] = all(r["equal_A"] for k, r in res.items() if isinstance(r, dict))
print("equal", res["equal"])
json.dump(res, open(OUT, "w"), indent=1)
