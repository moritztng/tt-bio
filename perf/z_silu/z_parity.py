#!/usr/bin/env python3
"""z-silu -- dump the fc1 output at the fold's own shape so the arms can be compared on host.

One arm per process (the arm is the JIT header). Deterministic input, production compute-kernel
config, production core grid. `--ref` also writes the torch fp32 reference for the same input.
"""
from __future__ import annotations
import argparse, json
import numpy as np, torch, ttnn

SHAPES = {"298": (1, 30, 298, 256), "512": (1, 16, 512, 256)}
N_OUT = 1024

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--shape", default="298", choices=list(SHAPES))
ap.add_argument("--out", required=True)
a = ap.parse_args()

shp = SHAPES[a.shape]
torch.manual_seed(0)
ta = torch.randn(*shp)
tb = torch.randn(256, N_OUT)

dev = ttnn.open_device(device_id=0)
try:
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    A = ttnn.from_torch(ta, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.L1_MEMORY_CONFIG)
    B = ttnn.from_torch(tb, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev,
                        memory_config=ttnn.DRAM_MEMORY_CONFIG)
    out = ttnn.linear(A, B, activation="silu", compute_kernel_config=cfg,
                      memory_config=ttnn.L1_MEMORY_CONFIG, dtype=ttnn.bfloat16,
                      core_grid=ttnn.CoreGrid(y=10, x=11))
    ttnn.synchronize_device(dev)
    t = ttnn.to_torch(out)
finally:
    ttnn.close_device(dev)

np.save(a.out, t.float().numpy())
print("wrote", a.out, tuple(t.shape), float(t.float().abs().mean()))
