#!/usr/bin/env python3
"""Per-rank cost of the transfer COUNT, not bytes, at trunk_step's real per-call sizes (M1: 516 fold reads of ~1.36 MB fp32,
580 rebuild uploads of ~0.61 MB bf16 per step) plus tiny-op dispatch rate. Run one per chip, concurrently.
    dispbw.py OUT.json START_EPOCH"""
import json, sys, time
import torch
from tt_bio import tenstorrent as T
import ttnn

out, start = sys.argv[1], float(sys.argv[2])
dev = T.get_device()
up = torch.randn(320, 1024).bfloat16()        # 0.66 MB bf16
dn = ttnn.from_torch(torch.randn(352, 1024), dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)   # 1.44 MB
x = ttnn.from_torch(torch.ones(32, 32).bfloat16(), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
ttnn.synchronize_device(dev)
time.sleep(max(0.0, start - time.time()))
res = {}
for rep in range(3):
    t0 = time.time()
    ts = [ttnn.from_torch(up, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(580)]
    ttnn.synchronize_device(dev)
    t1 = time.time()
    for t in ts:
        ttnn.deallocate(t)
    for _ in range(516):
        ttnn.to_torch(dn)
    t2 = time.time()
    y = x
    for _ in range(3000):
        z = ttnn.add(y, 1.0)
        if y is not x:
            ttnn.deallocate(y)
        y = z
    ttnn.synchronize_device(dev)
    t3 = time.time()
    res[rep] = dict(up580_s=round(t1 - t0, 3), down516_s=round(t2 - t1, 3), ops3000_s=round(t3 - t2, 3))
res["t_end"] = time.time()
json.dump(res, open(out, "w"))
print(json.dumps(res))
