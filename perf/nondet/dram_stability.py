#!/usr/bin/env python3
"""Zero-compute host<->device transfer stability probe.

Writes a fixed pattern to device DRAM and checks three things per iteration, none of
which involve a compute kernel:

  read-read      the SAME device tensor read back twice          -> readback path
  read-written   the readback vs the bytes we uploaded           -> either direction
  copy-copy      two SEPARATE uploads of the same host bytes     -> upload path

That splits a transfer-path fault from a racy kernel. dtype matters here: the trunk moves
bf16 and the Protenix-v2 diffusion moves fp32 (PROTENIX_DIFFUSION_FP32_DEVICE=1), so the
precision argument is the same A/B the pair-cond probe runs.

Usage: dram_stability.py [mb_per_iter] [iters] [bf16|fp32]
"""
import sys

import torch
import ttnn

mb = int(sys.argv[1]) if len(sys.argv) > 1 else 2048
iters = int(sys.argv[2]) if len(sys.argv) > 2 else 6
prec = sys.argv[3] if len(sys.argv) > 3 else "bf16"
dt = ttnn.float32 if prec == "fp32" else ttnn.bfloat16
itemsize = 4 if prec == "fp32" else 2

dev = ttnn.open_device(device_id=0)
g = torch.Generator().manual_seed(0)
bad = 0
try:
    for i in range(iters):
        n = mb * 1024 * 1024 // itemsize
        host = torch.randint(-32768, 32767, (n,), dtype=torch.int32, generator=g).to(torch.bfloat16)
        if prec == "fp32":
            host = host.to(torch.float32)
        t = ttnn.from_torch(host, device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt)
        ttnn.synchronize_device(dev)
        a = ttnn.to_torch(t)
        b = ttnn.to_torch(t)
        t2 = ttnn.from_torch(host, device=dev, layout=ttnn.TILE_LAYOUT, dtype=dt)
        ttnn.synchronize_device(dev)
        c = ttnn.to_torch(t2)
        rr = int((a != b).sum())
        rw = int((a != host.to(a.dtype)).sum())
        cc = int((a != c).sum())
        bad += rr + rw + cc
        print(f"iter {i}: {mb} MB {prec}  read-read={rr}  read-written={rw}  copy-copy={cc}",
              flush=True)
        del t, t2
finally:
    ttnn.close_device(dev)
print(f"TOTAL mismatched elements: {bad} over {iters}x{mb} MB {prec}")
sys.exit(1 if bad else 0)
