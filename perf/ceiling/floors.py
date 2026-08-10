#!/usr/bin/env python3
"""Three terms the arithmetic ceiling needs and does not have yet.

1. The per-ttnn-call floor. The diffusion stage issues 202,116 calls in 9.07 s. If a call cannot
   cost less than t0 no matter how small it is, then 202,116 * t0 is a hard floor on that stage
   and no amount of arithmetic optimisation touches it.
2. The Pairformer block's weight bytes, counted off the real module, not guessed.
3. A DRAM read roof measured on a ladder that has actually flattened. W1 showed the 8-64 MB
   ladder is still climbing 3.4% at the top, so 64 MB is not a roof.
"""
import json
import statistics as st
import sys
import time

import torch
import ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
dev = get_device()
g = dev.compute_with_storage_grid_size()
res = {"grid": [g.x, g.y]}
torch.manual_seed(0)

# ---- 1. per-call floor -------------------------------------------------------------------
print("== per-ttnn-call floor ==", flush=True)
tiny = ttnn.from_torch(torch.randn(1, 1, 32, 32), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
small = ttnn.from_torch(torch.randn(1, 1, 320, 320), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
percall = {}
for lbl, fn in (
    ("add 1 tile (L1)", lambda: ttnn.deallocate(ttnn.add(tiny, tiny, memory_config=L1))),
    ("mul_ in place 1 tile", lambda: ttnn.multiply_(tiny, tiny)),
    ("add 100 tiles (L1)", lambda: ttnn.deallocate(ttnn.add(small, small, memory_config=L1))),
    ("matmul 1 tile (L1)", lambda: ttnn.deallocate(ttnn.matmul(tiny, tiny, memory_config=L1))),
):
    for _ in range(50):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(5):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(500):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) * 1e6 / 500)
    percall[lbl] = round(st.median(o), 3)
    print(f"  {lbl:24s} {st.median(o):8.3f} us/call", flush=True)
res["per_call_us"] = percall
ttnn.deallocate(tiny)
ttnn.deallocate(small)

# ---- 3. DRAM roofs on a ladder that flattens --------------------------------------------
print("== DRAM roofs, ladder to 192 MB ==", flush=True)
rd, wr = {}, {}
for mb in (32, 64, 96, 128, 160, 192):
    n = mb * 1_000_000 // 2 // 320
    t = ttnn.from_torch(torch.randn(1, 1, ((n + 31) // 32) * 32, 320), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    byt = t.volume() * 2
    # DRAM -> DRAM clone: reads byt, writes byt
    for _ in range(3):
        ttnn.deallocate(ttnn.clone(t, memory_config=DRAM))
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(5):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(3):
            ttnn.deallocate(ttnn.clone(t, memory_config=DRAM))
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / 3)
    s = st.median(o)
    rd[byt // 1_000_000] = round(byt / 1e9 / s, 1)  # one-way read GB/s at the mixed r+w point
    print(f"  {byt/1e6:7.1f} MB  DRAM->DRAM {s*1e3:8.3f} ms  read {byt/1e9/s:7.1f} GB/s  write {byt/1e9/s:7.1f} GB/s (mixed)", flush=True)
    ttnn.deallocate(t)
res["dram_dram_mixed_GBs_each_way"] = rd
json.dump(res, open(sys.argv[1], "w"), indent=2)
print("wrote", sys.argv[1], flush=True)
