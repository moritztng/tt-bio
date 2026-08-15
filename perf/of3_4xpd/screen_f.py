#!/usr/bin/env python3
"""Screen lever F before anyone designs it: what is the fp32-softmax tail actually worth?

F would fuse the four programs between TriangleAttention's two matmuls -- typecast to
fp32, add the scaled bias, softmax, typecast back to bf16 -- into one `generic_op` over
the L1 shard. Its 2.0-3.7 s band was DERIVED, never measured, and it is the only lever
left whose band covers the 1.04 s still owed for 4.000x.

This measures the tail as it runs today at the production block geometry, and the floor a
perfect fusion could reach: the softmax reduction alone, which no fusion can delete. The
difference times the blocks in a fold is F's ceiling.

Timed a whole call's worth of blocks per iteration, never one op with a sync around it
(`tt-bio-isolated-op-timing-oversync-inflates-cost`).
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T

# The production TriangleAttention block: q/k/v [rows, heads, S, dh], scores
# [rows, heads, S, S]. The census reports 43 blocks per call at rows=12.
ROWS, HEADS, S, DH = 12, 4, 512, 32
BLOCKS_PER_CALL = 43
CALLS_PER_FOLD = 440           # measured: FP32_SOFTMAX_STATS calls/3 folds on this branch
ITERS = 3


def main():
    dev = T.get_device()
    out = {"rows": ROWS, "heads": HEADS, "seq": S,
           "blocks_per_call": BLOCKS_PER_CALL, "calls_per_fold": CALLS_PER_FOLD}
    height_per_row = HEADS * S
    shard = T._fp32_softmax_shard(ROWS, height_per_row, S)
    assert shard is not None, "no shard config at the production geometry"

    sc = ttnn.from_torch(torch.randn(ROWS, HEADS, S, S, dtype=torch.bfloat16),
                         layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    bias = ttnn.from_torch(torch.randn(1, HEADS, S, S, dtype=torch.bfloat16),
                           layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    scale_inv = 1.0 / (DH ** 0.5)

    def tail_full():
        o = T._fp32_softmax_tail(sc, bias, scale_inv, None, shard)
        ttnn.deallocate(o)

    def softmax_only():
        """The irreducible middle: the fp32 reduction over the same sharded block."""
        l1 = ttnn.to_memory_config(sc, shard)
        f32 = ttnn.typecast(l1, ttnn.float32, memory_config=shard)
        ttnn.deallocate(l1)
        a = ttnn.softmax_in_place(f32)
        ttnn.deallocate(a)

    def softmax_bare():
        """Softmax with the block already fp32-resident: no cast, no move, no write-back."""
        f32 = ttnn.typecast(ttnn.to_memory_config(sc, shard), ttnn.float32, memory_config=shard)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        a = ttnn.softmax_in_place(f32)
        ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        ttnn.deallocate(a)
        return dt

    for name, fn in (("tail_today", tail_full), ("shard+cast+softmax", softmax_only)):
        fn(); ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        out[name + "_ms_per_block"] = round(ms, 4)
        print(f"{name:22s} {ms:8.4f} ms per block", flush=True)

    bare = min(softmax_bare() for _ in range(3)) * 1e3
    out["softmax_alone_ms_per_block"] = round(bare, 4)
    print(f"{'softmax alone':22s} {bare:8.4f} ms per block", flush=True)

    per_fold = BLOCKS_PER_CALL * CALLS_PER_FOLD
    out["blocks_per_fold"] = per_fold
    out["tail_s_per_fold"] = round(out["tail_today_ms_per_block"] * per_fold / 1e3, 3)
    out["floor_s_per_fold"] = round(out["softmax_alone_ms_per_block"] * per_fold / 1e3, 3)
    out["F_ceiling_s"] = round(out["tail_s_per_fold"] - out["floor_s_per_fold"], 3)
    print(f"\nblocks per fold {per_fold}", flush=True)
    print(f"tail as it runs today   {out['tail_s_per_fold']:7.3f} s per fold", flush=True)
    print(f"softmax-alone floor     {out['floor_s_per_fold']:7.3f} s per fold", flush=True)
    print(f"F ceiling               {out['F_ceiling_s']:7.3f} s per fold", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    T.cleanup()


if __name__ == "__main__":
    main()
