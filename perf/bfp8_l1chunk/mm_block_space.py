#!/usr/bin/env python3
"""Is the minimal_matmul block table L1-bound? Enumerate the legal (M,K,N) block space per
_MM_BLOCK key at bf16 and at bfp8 and compare the two sets.

The reopened thin-K question is whether halving the tile lets more work sit in one tile-block.
That can only be true if L1 is what caps the block today. mm_generic.build allocates exactly
four CBs, so the budget is arithmetic and needs no device.
"""
import json

import ttnn

from tt_bio.mm_generic import tile_bytes
from tt_bio.sdpa_generic import L1_PER_CORE, PROGRAM_RESERVE
import tt_bio.tenstorrent as TT

BUDGET = L1_PER_CORE - PROGRAM_RESERVE


def cb_l1(M_block, K_block, N_block, dtype, fp32_dest_acc=True):
    """mm_generic.build's four CBs: in0 x2, in1 x2, out x2, interm x1 (fp32 under acc)."""
    t = tile_bytes(dtype)
    interm = tile_bytes(ttnn.float32 if fp32_dest_acc else ttnn.bfloat16)
    return (M_block * K_block * 2 * t + K_block * N_block * 2 * t
            + M_block * N_block * 2 * t + M_block * N_block * interm)


def legal(kt, nt, dtype, mmax=64):
    """Every (M_block, K_block, N_block) that fits, with K_block == kt (the table's invariant,
    which is what keeps every entry bit-exact) and N_block dividing nt."""
    out = set()
    for M_block in range(1, mmax + 1):
        for N_block in range(1, nt + 1):
            if nt % N_block:
                continue
            if cb_l1(M_block, kt, N_block, dtype) <= BUDGET:
                out.add((M_block, kt, N_block))
    return out


def main():
    res = {"budget": BUDGET, "keys": {}}
    for (kt, nt), cfg in sorted(TT._MM_BLOCK.items()):
        s16, s8 = legal(kt, nt, ttnn.bfloat16), legal(kt, nt, ttnn.bfloat8_b)
        shipped = (cfg[0], cfg[1], cfg[2])
        res["keys"][f"{kt},{nt}"] = {
            "shipped": list(cfg),
            "shipped_l1_bf16": cb_l1(*shipped, ttnn.bfloat16),
            "shipped_l1_bfp8": cb_l1(*shipped, ttnn.bfloat8_b),
            "n_legal_bf16": len(s16), "n_legal_bfp8": len(s8),
            "newly_legal": sorted(s8 - s16),
            "max_M_bf16": max(m for m, _k, _n in s16),
            "max_M_bfp8": max(m for m, _k, _n in s8),
        }
        r = res["keys"][f"{kt},{nt}"]
        print(f"key (kt={kt:2d}, nt={nt:2d})  shipped {cfg}  "
              f"L1 {r['shipped_l1_bf16']:7d} B ({100*r['shipped_l1_bf16']/BUDGET:5.1f}% of budget)"
              f" -> bfp8 {r['shipped_l1_bfp8']:7d} B  |  legal configs {r['n_legal_bf16']:3d} -> "
              f"{r['n_legal_bfp8']:3d},  max M_block {r['max_M_bf16']} -> {r['max_M_bfp8']}")
    with open("perf/bfp8_l1chunk/mm_block_space.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
