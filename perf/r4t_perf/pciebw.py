#!/usr/bin/env python3
"""Host<->device bandwidth of one rank, the transfer shapes trunk_step moves (M1): bf16 tile uploads of TENSOR_MB each,
fp32 reads of the same. Run one per chip, concurrently, to price bytes under root-complex sharing.
    pciebw.py OUT.json [TOTAL_MB=512] [TENSOR_MB=1] [START_EPOCH]
(all ranks start transferring at START_EPOCH, after their device opens)"""
import json, sys, time
import torch
from tt_bio import tenstorrent as T
import ttnn

out, total, each = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 512, float(sys.argv[3]) if len(sys.argv) > 3 else 1
dev = T.get_device()
n_up = int(each * 2**20 / 2)          # bf16 elements per tensor
rows = max(32, (n_up // 1024) // 32 * 32)
x16 = torch.randn(rows, 1024).bfloat16()
x32 = torch.randn(rows, 1024)
k = max(1, int(total / each))
if len(sys.argv) > 4:
    time.sleep(max(0.0, float(sys.argv[4]) - time.time()))
res = {}
for rep in range(3):
    ttnn.synchronize_device(dev)
    t0 = time.time()
    ts = [ttnn.from_torch(x16, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) for _ in range(k)]
    ttnn.synchronize_device(dev)
    t1 = time.time()
    for t in ts:
        ttnn.deallocate(t)
    d = ttnn.from_torch(x32, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
    ttnn.synchronize_device(dev)
    t2 = time.time()
    for _ in range(k // 2):
        ttnn.to_torch(d)
    t3 = time.time()
    up_B, dn_B = k * x16.numel() * 2, (k // 2) * x32.numel() * 4
    res[rep] = dict(up_GBs=round(up_B / (t1 - t0) / 1e9, 3), down_GBs=round(dn_B / (t3 - t2) / 1e9, 3),
                    up_s=round(t1 - t0, 3), down_s=round(t3 - t2, 3), up_MB=round(up_B / 1e6), down_MB=round(dn_B / 1e6))
    ttnn.deallocate(d)
res["t_end"] = time.time()
json.dump(res, open(out, "w"))
print(json.dumps(res))
