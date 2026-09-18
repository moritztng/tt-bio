#!/usr/bin/env python3
"""Validate perf/bfp8_qkv/l1price.py's replication against the LIVE _tri_att_qkv_l1_config.

No device is opened: `_matmul_cb_budget` is patched to the two candidate constants, which is the
only thing in that function that would have reached for hardware. The point is that the sweep's
accept/reject is the SHIPPED predicate's, not my transcription of it -- a re-derived gate that
disagrees with the real one prices nothing.
"""
import sys
sys.path.insert(0, "/home/ttuser/.coworker/wt/bfp8-qkv-matmul")
sys.path.insert(0, "/home/ttuser/.coworker/wt/bfp8-qkv-matmul/perf/bfp8_qkv")

import l1price
from tt_bio import tenstorrent as T

print("live COMPUTE_GRID_MAIN =", tuple(T.COMPUTE_GRID_MAIN),
      "| replication GRID =", l1price.GRID)
print("live _MATMUL_CB_SLACK =", T._MATMUL_CB_SLACK, "| replication =", l1price.SLACK)
print("live _FP32_ACC_TILE  =", T._FP32_ACC_TILE, "| replication =", l1price.FP32_ACC_TILE)
assert tuple(T.COMPUTE_GRID_MAIN) == l1price.GRID
assert T._MATMUL_CB_SLACK == l1price.SLACK and T._FP32_ACC_TILE == l1price.FP32_ACC_TILE

# bfp8 cannot be expressed as an int elem_bytes, which is the whole trap. The live function takes
# elem_bytes, so only the bf16 arm (elem_bytes=2) is directly comparable; that is exactly the arm
# that decides whether the reclaim moves a bucket, since the bfp8 arm is the proposed change.
agree = disagree = 0
for bname, l1 in l1price.BUDGETS.items():
    T._matmul_cb_budget = lambda _l1=l1: _l1
    for gname, cz, wqkv in l1price.GEOMS:
        for tok in l1price.TOKENS:
            m = tok * tok
            if m % 32:
                continue
            mt, kt, nt = m // 32, cz // 32, wqkv // 32
            T._tri_att_qkv_l1_config.cache_clear()
            live = T._tri_att_qkv_l1_config(mt, kt, nt, 2) is not None
            mine, why, _ = l1price.fits(mt, kt, nt, 2048, l1)
            tag = "OK " if live == mine else "MISMATCH"
            if live == mine:
                agree += 1
            else:
                disagree += 1
                print("  %s %s tok=%d m_tiles=%d live=%s mine=%s (%s)"
                      % (tag, bname, tok, mt, live, mine, why))
print("\nreplication vs live at elem_bytes=2: %d agree, %d disagree" % (agree, disagree))

# And the shipped in0_block_w / per_core_M for the buckets that DO fit, off the live object.
T._matmul_cb_budget = lambda: 1461760
print("\nlive accepted plans at the allocator budget (elem_bytes=2):")
for gname, cz, wqkv in l1price.GEOMS:
    for tok in l1price.TOKENS:
        m = tok * tok
        if m % 32:
            continue
        T._tri_att_qkv_l1_config.cache_clear()
        c = T._tri_att_qkv_l1_config(m // 32, cz // 32, wqkv // 32, 2)
        if c is not None:
            print("  %-28s tok=%-4d in0_block_w=%d per_core_M=%d per_core_N=%d"
                  % (gname, tok, c.in0_block_w, c.per_core_M, c.per_core_N))
