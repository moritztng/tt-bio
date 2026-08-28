#!/usr/bin/env python3
"""Standalone repro attempt for the protenix-v1 512 aa wedge: bare ttnn, no tt_bio model.

The wedging op is `linear_no_bias_z` in tt_bio/protenix.py::_diffusion_pair_cond, a
ttnn.linear on CORE_GRID_MAIN. At 512 tokens
protenix-v1 runs (N*N, 256) x (256, 128) and hangs intermittently; protenix-v2 runs
(N*N, 512) x (512, 256) on the same line and does not. This drives just that matmul in a loop
at both shapes so the failure can be attributed to ttnn without tt-bio in the picture.

    TT_VISIBLE_DEVICES=0 ... python3 /tmp/pv1/repro.py <iters>
"""
import sys, time
import ttnn
import torch

ITERS = int(sys.argv[1]) if len(sys.argv) > 1 else 40
N = 512
dev = ttnn.open_device(device_id=0)
# The grid tt_bio uses for this op: CORE_GRID_MAIN = CoreGrid(y=COMPUTE_GRID_Y, x=COMPUTE_GRID_X_11)
from tt_bio.tenstorrent import CORE_GRID_MAIN, COMPUTE_GRID_MAIN  # constants only, no model
print(f"grid={COMPUTE_GRID_MAIN} iters={ITERS} N={N}", flush=True)

BALLAST_GIB = float(__import__("os").environ.get("REPRO_BALLAST_GIB", "0"))
_ballast = []
if BALLAST_GIB:
    # The real fold holds ~1.2-1.7 GiB of live trunk tensors when it reaches this matmul,
    # while a bare repro reaches it on an almost-empty device. Occupy DRAM first so the
    # allocator is in a comparable state.
    per = 256 * 2 ** 20  # 256 MiB per bf16 block
    n = int(BALLAST_GIB * 2 ** 30 // per)
    for _ in range(n):
        _ballast.append(ttnn.from_torch(torch.zeros(1, per // 2 // 1024, 1024),
                                        layout=ttnn.TILE_LAYOUT, device=dev,
                                        dtype=ttnn.bfloat16))
    print(f"ballast: {n} x 256 MiB = {n * 0.25:.2f} GiB resident", flush=True)


def run(k, out, label):
    zc = ttnn.from_torch(torch.randn(1, N * N, k), layout=ttnn.TILE_LAYOUT,
                         device=dev, dtype=ttnn.bfloat16)
    w = ttnn.from_torch(torch.randn(k, out), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16)
    for i in range(ITERS):
        t0 = time.time()
        pz = ttnn.linear(zc, w, dtype=ttnn.bfloat16, core_grid=CORE_GRID_MAIN)
        ttnn.synchronize_device(dev)
        dt = time.time() - t0
        print(f"{label} iter {i:3d}  {dt*1e3:8.1f} ms", flush=True)
        ttnn.deallocate(pz)
    ttnn.deallocate(zc); ttnn.deallocate(w)

try:
    run(256, 128, "v1-shape (262144,256)x(256,128)")
    run(512, 256, "v2-shape (262144,512)x(512,256)")
finally:
    ttnn.close_device(dev)
print("REPRO LOOP COMPLETED", flush=True)
