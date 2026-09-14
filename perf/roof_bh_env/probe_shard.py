#!/usr/bin/env python3
"""Why the tuned 8x8 rectangle refuses on Wormhole while the same 64 cores as a CoreRangeSet do not."""
import json, sys
import torch, ttnn
import tt_bio.tenstorrent as T

dev = T.get_device()
cc = dev.compute_with_storage_grid_size()
print("grid x=%d y=%d  COMPUTE_GRID_MAIN=%s  IS_SMALL_GRID=%s" % (cc.x, cc.y, T.COMPUTE_GRID_MAIN, T._IS_SMALL_GRID))
print("L1_TOTAL fallback=%s  live l1 total=%s" % (T.L1_TOTAL_BYTES_WORMHOLE, T._l1_total_bytes()))
S, heads = 512, 4
height_per_row = heads * S
for grid in [(8, 8), (9, 8)]:
    T._FP32_SOFTMAX_L1_GRID = grid
    T._fp32_softmax_core_grid.cache_clear()
    cg = T._fp32_softmax_core_grid(64)
    sh = T._fp32_softmax_shard(12, height_per_row, S, 64)
    print("\ngrid=%s  core_grid_obj=%r" % (grid, cg))
    print("  shard cfg: %s" % sh)
T.cleanup()
