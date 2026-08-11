#!/usr/bin/env python3
"""triatt-engine-extract-land: torch.equal spot check for the two extracted engine fixes.

The fold A/B (perf/size512/fold_ab512.py, arm `prev`) is the parity gate; this script is the
per-op cross-check on the exact shapes the levers touch, in one process, arms flipped at
runtime:

1. `_pair_proj_minimal_matmul`: `_pair_proj_linear` with `_PAIR_PROJ_MM` off (the stock
   `ttnn.linear` DRAM path) vs on (the minimal_matmul leg) at K=256, N=256, for the production
   pair sizes. Must be torch.equal per size.
2. `_MM_BLOCK[8]`: `minimal_matmul` via `_qkv_mm_config` with the old (2,8,1,2,1) vs new
   (4,8,1,4,1) entry at the same shapes. Same K_block in both, so the accumulation order is
   identical and the outputs must be torch.equal.

Writes perf/triatt_opt/mm_extract_check.json.
"""
import json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch
import ttnn
from tt_bio import tenstorrent as T

RES = {"ttnn": "0.68.0", "note": "pc card 0 (p150a); qb2 card 2 numbers live in stage1_sweep.json"}


def timed(fn, warm=3, reps=5):
    import statistics as st
    dev = T.get_device()
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        del r
    return st.median(ts)


def mk(dev, shape):
    t = torch.randn(*shape, dtype=torch.float32).to(torch.bfloat16)
    return ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                           memory_config=ttnn.DRAM_MEMORY_CONFIG)


def main():
    dev = T.get_device()
    ckc = ttnn.WormholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=False, packer_l1_acc=True)
    RES["grid"] = list(T.COMPUTE_GRID_MAIN)
    rows = []
    for S in (298, 320, 384, 512, 576, 640):
        x = mk(dev, (S, S, 256))
        w = mk(dev, (256, 256))
        row = {"S": S}

        # Lever 1: the pair-projection minimal_matmul leg vs the stock ttnn.linear path.
        T._PAIR_PROJ_MM = False
        old = T._pair_proj_linear(x, w, ckc, ttnn.bfloat16)
        row["linear_ms"] = timed(lambda: T._pair_proj_linear(x, w, ckc, ttnn.bfloat16)) * 1e3
        T._PAIR_PROJ_MM = True
        new = T._pair_proj_linear(x, w, ckc, ttnn.bfloat16)
        row["mm_ms"] = timed(lambda: T._pair_proj_linear(x, w, ckc, ttnn.bfloat16)) * 1e3
        row["pair_proj_eq"] = bool(torch.equal(ttnn.to_torch(old), ttnn.to_torch(new)))
        row["pair_proj_ratio"] = row["linear_ms"] / row["mm_ms"]
        ttnn.deallocate(old)
        ttnn.deallocate(new)

        # Lever 2: _MM_BLOCK[8] old vs new through the shipped config reader.
        T._MM_BLOCK[8] = (2, 8, 1, 2, 1)
        cfg2 = T._qkv_mm_config(x, w)
        o2 = ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg2)
        row["blk2_ms"] = timed(lambda: ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg2)) * 1e3
        T._MM_BLOCK[8] = (4, 8, 1, 4, 1)
        cfg4 = T._qkv_mm_config(x, w)
        o4 = ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg4)
        row["blk4_ms"] = timed(lambda: ttnn.experimental.minimal_matmul(
            input_tensor=x, weight_tensor=w, compute_kernel_config=ckc,
            dtype=ttnn.bfloat16, config=cfg4)) * 1e3
        row["mm_block_eq"] = bool(torch.equal(ttnn.to_torch(o2), ttnn.to_torch(o4)))
        row["mm_block_ratio"] = row["blk2_ms"] / row["blk4_ms"]
        for t in (o2, o4, x, w):
            ttnn.deallocate(t)
        rows.append(row)
        print("SIZE", json.dumps(row), flush=True)

    RES["per_size"] = rows
    RES["all_bit_exact"] = all(r["pair_proj_eq"] and r["mm_block_eq"] for r in rows)
    out = REPO / "perf" / "triatt_opt" / "mm_extract_check.json"
    out.write_text(json.dumps(RES, indent=1))
    print("ALL_BIT_EXACT", RES["all_bit_exact"], flush=True)
    print("wrote", out, flush=True)


if __name__ == "__main__":
    main()
