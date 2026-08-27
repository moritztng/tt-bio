#!/usr/bin/env python3
"""Bare ttnn matmul at protenix-v1's pair-cond shapes, under the tt-metal watcher.

The watcher pinned the wedge to ttnn's multicast matmul (in0 mcast sender on a NOC semaphore
wait, in1 receivers on a CB reserve-back wait) -- see perf/pxv1/hang_watcher_capture.txt. If a
bare matmul at v1's shapes wedges under the watcher while v2's does not, that is a
self-contained ttnn reproducer with no tt-bio in it, fileable upstream.

An earlier bare-matmul attempt found nothing, but it ran in bfloat16 while the real op is
FLOAT32, without the watcher, and for only 25 iterations. This fixes all three.

One shape per process, because the watcher SIGABRTs on detection and would otherwise hide which
shape did it.

    TT_METAL_WATCHER=10 TT_VISIBLE_DEVICES=0 ... \
      python3 scripts/protenix_v1_port/repro_mm_watcher_sweep.py <K> <N_out> [iters]

Shapes of interest at 512 tokens (M = 512*512 = 262144 rows):
    K=256 N=128   protenix-v1   K 8 tiles, out 4 tiles   <- hangs in the real fold
    K=512 N=256   protenix-v2   K 16 tiles, out 8 tiles  <- 7/7 clean in the real fold
    K=256 N=256   isolates the output width
    K=512 N=128   isolates K
"""
import sys
import time

import torch
import ttnn

from tt_bio.tenstorrent import CORE_GRID_MAIN, COMPUTE_GRID_MAIN

M = 512 * 512


def main(k, n_out, iters):
    dev = ttnn.open_device(device_id=0)
    print(f"grid={COMPUTE_GRID_MAIN} M={M} K={k} N_out={n_out} "
          f"(K={k // 32} tiles, out={n_out // 32} tiles) dtype=float32 iters={iters}",
          flush=True)
    try:
        zc = ttnn.from_torch(torch.randn(1, M, k), layout=ttnn.TILE_LAYOUT,
                             device=dev, dtype=ttnn.float32)
        w = ttnn.from_torch(torch.randn(k, n_out), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.float32)
        for i in range(iters):
            t0 = time.time()
            pz = ttnn.linear(zc, w, dtype=ttnn.float32, core_grid=CORE_GRID_MAIN)
            ttnn.synchronize_device(dev)
            print(f"iter {i:3d}  {(time.time() - t0) * 1e3:8.1f} ms", flush=True)
            ttnn.deallocate(pz)
        ttnn.deallocate(zc)
        ttnn.deallocate(w)
    finally:
        ttnn.close_device(dev)
    print(f"SWEEP OK K={k} N={n_out}", flush=True)
    return 0


if __name__ == "__main__":
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    IT = int(sys.argv[3]) if len(sys.argv) > 3 else 30
    sys.exit(main(K, N, IT))
