#!/usr/bin/env python3
"""Which leading-dim block the fp32-softmax planner picks per token count, and what that
leaves the triangle-attention score tensor costing.

This is the whole OpenFold3 576 cap in one table. `_fp32_softmax_l1_rows` needs
`blk * heads * q_len` to be a multiple of `cores * 32`, and `q_len` is the LOGICAL token
count, so at a ragged N no affordable block height divides and the plan goes dark. With no
plan the only bound left is the fixed `_FP32_SOFTMAX_BLOCK_BYTES`, which at these sizes
never fires -- so a ragged N materialises the whole `O(N^3 x heads)` score tensor and an
aligned N materialises ~40 MB.

Every function called here is host arithmetic, so it needs no device. It DOES need the
tree's grid-scaled constants, which `_apply_grid_thresholds` rewrites on device open --
run it on the part whose answer you want, not on a p150a when you mean a Galaxy.

    python3 perf/ceiling_of3/l1_plan_probe.py <tree>
"""
from __future__ import annotations

import sys

sys.path.insert(0, sys.argv[1])
import tt_bio.tenstorrent as T  # noqa: E402

HEADS = 4  # OF3 msa_module pair_stack: no_heads_pair=4 (openfold3_msa_embedder.py _MSA_TRI_DIMS)
SIZES = [512, 528, 544, 560, 576, 592, 608, 614, 640, 641, 672, 704, 736, 745, 760, 768,
         800, 832, 864, 896, 928, 960, 992, 1024]


def main() -> None:
    print(f"grid={T.COMPUTE_GRID_MAIN} l1_grid={T._FP32_SOFTMAX_L1_GRID} "
          f"bytes_per_core={T._FP32_SOFTMAX_L1_BYTES_PER_CORE} "
          f"core_cap={T._FP32_SOFTMAX_L1_CORE_CAP} "
          f"block_bytes={T._FP32_SOFTMAX_BLOCK_BYTES} "
          f"any_cores={T._FP32_SOFTMAX_L1_ANY_CORES} "
          f"float_cores={T._FP32_SOFTMAX_L1_FLOAT_CORES}")
    print(f"{'N':>5} {'ragged':>6} {'tuned':>5} {'rows':>5} {'cores':>5} {'cap_blk':>7} "
          f"{'blk':>5} {'score_GB':>9}")
    for n in SIZES:
        height_per_row = HEADS * n
        per_row = height_per_row * n * 4          # one fp32 score row
        tuned = T._fp32_softmax_l1_rows(per_row, height_per_row, None)
        rows, cores = T._fp32_softmax_l1_plan(per_row, height_per_row, n, None, None, None)
        cap_blk = max(32, int(T._FP32_SOFTMAX_BLOCK_BYTES // per_row) // 32 * 32)
        blk = min(cap_blk, rows) if rows else cap_blk
        blk = min(blk, n)
        score = blk * height_per_row * n * 2      # the bf16 q@k^T the block allocates
        print(f"{n:>5} {'yes' if n % 32 else 'no':>6} {tuned:>5} {rows:>5} {cores:>5} "
              f"{cap_blk:>7} {blk:>5} {score / 1e9:>9.3f}")


if __name__ == "__main__":
    main()
