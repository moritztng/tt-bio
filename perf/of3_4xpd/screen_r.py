#!/usr/bin/env python3
"""Lever R screen: can a row-block slice land straight in a height-sharded destination?

The fp32-softmax row-block loop slices q/k/v per block out of DRAM, shards the scores
inside the tail, and concatenates the blocks back at the end. R would slice straight into
the sharded destination and skip a DRAM round trip. Two questions, in order:

  1. does `ttnn.slice` accept a sharded `memory_config` at all? If not, R is dead --
     `ttnn.slice` is never a view, so the offset-write plan is already refuted.
  2. is it faster than the slice + `to_memory_config` pair it replaces? Kill under 1.10x.

Timed a whole call's worth of blocks per iteration (43 blocks), not one slice with a sync
around it: isolated per-op timing oversyncs and inflates the cost about 2x
(`tt-bio-isolated-op-timing-oversync-inflates-cost`).
"""
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch
import ttnn
import tt_bio.tenstorrent as T

ROWS, HEADS, SEQ, HD = 512, 4, 512, 32   # production TriangleAttention q/k/v at 512 aa
BLK = 12                                  # the L1 row block the census reports (43 per call)
ITERS = 5


def main():
    dev = T.get_device()
    q = ttnn.from_torch(torch.randn(ROWS, HEADS, SEQ, HD, dtype=torch.bfloat16),
                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    shard = T._fp32_softmax_shard(BLK, HEADS * SEQ, HD)
    out = {"shape": [ROWS, HEADS, SEQ, HD], "blk": BLK,
           "blocks_per_call": (ROWS + BLK - 1) // BLK,
           "shard_config": str(shard)}
    print("shard config:", shard, flush=True)

    starts = [(s, min(s + BLK, ROWS)) for s in range(0, ROWS, BLK)]
    starts = [(s, e) for s, e in starts if e - s == BLK]     # whole blocks only

    def sliced_then_sharded():
        for s, e in starts:
            x = ttnn.slice(q, [s, 0, 0, 0], [e, HEADS, SEQ, HD])
            y = ttnn.to_memory_config(x, shard)
            ttnn.deallocate(x)
            ttnn.deallocate(y)

    def sliced_into_shard():
        for s, e in starts:
            y = ttnn.slice(q, [s, 0, 0, 0], [e, HEADS, SEQ, HD], memory_config=shard)
            ttnn.deallocate(y)

    try:
        sliced_into_shard()
        out["slice_accepts_sharded_memory_config"] = True
    except Exception as exc:                                  # noqa: BLE001
        out["slice_accepts_sharded_memory_config"] = False
        out["refusal"] = f"{type(exc).__name__}: {exc}"[:400]
        print("K-R FIRES: ttnn.slice refused the sharded memory_config\n", out["refusal"], flush=True)
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
        T.cleanup()
        return

    for name, fn in (("slice+to_memory_config", sliced_then_sharded),
                     ("slice_into_shard", sliced_into_shard)):
        fn(); ttnn.synchronize_device(dev)                    # warm the programs
        t0 = time.perf_counter()
        for _ in range(ITERS):
            fn()
        ttnn.synchronize_device(dev)
        ms = (time.perf_counter() - t0) * 1e3 / ITERS
        out[name + "_ms_per_call"] = round(ms, 4)
        print(f"{name:24s} {ms:8.4f} ms per call ({len(starts)} blocks)", flush=True)

    out["ratio"] = round(out["slice+to_memory_config_ms_per_call"]
                         / out["slice_into_shard_ms_per_call"], 4)
    print(f"ratio {out['ratio']:.4f}x  (kill under 1.10x)", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    T.cleanup()


if __name__ == "__main__":
    main()
