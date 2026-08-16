#!/usr/bin/env python3
"""DRAM readback stability probe: fill device DRAM with a fixed pattern, read it back
twice, count bit flips between reads. A marginal cell/timing path shows up as a
mismatch between two reads of the same written data. Zero device compute, pure I/O.
Usage: dram_stability.py [mb_per_iter] [iters]
"""
import sys

import torch
import ttnn

mb = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 6

dev = ttnn.open_device(device_id=0)
g = torch.Generator().manual_seed(0)
flips_total = 0
try:
    for i in range(iters):
        n = mb * 1024 * 1024 // 2
        host = torch.randint(-32768, 32767, (n,), dtype=torch.int32, generator=g).to(torch.bfloat16)
        t = ttnn.from_torch(host, device=dev, layout=ttnn.TILE_LAYOUT)
        ttnn.synchronize_device(dev)
        a = ttnn.to_torch(t)
        b = ttnn.to_torch(t)
        flips = int((a != b).sum())
        match_written = int((a != host).sum())
        flips_total += flips
        print(f"iter {i}: {mb} MB, read1-vs-read2 flips={flips}, read1-vs-written={match_written}", flush=True)
        del t
finally:
    ttnn.close_device(dev)
print(f"TOTAL read-read flips: {flips_total} over {iters}x{mb} MB")
sys.exit(1 if flips_total else 0)
