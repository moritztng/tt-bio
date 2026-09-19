#!/usr/bin/env python3
"""Does the reordered reader's bigger OUT_CB survive the allocator at 512 aa, and is it still exact?

`trix-transaction` measured the fix through an XOR proxy that reads the wrong pages. The legal form
reads the same pages in a different order, and the price is the writer's window: it gathers along
the row index, so with S channel tiles streamed back to back it holds 32*S tiles instead of 32.
This walks the (S, buffers) ladder, reports the CB bytes the allocator was asked for, whether the
program built and ran, and `torch.equal` against `ttnn.permute` for every rung that did.

The allocator answer is the gate on T1 and it is settled here before anything is timed.
"""
from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import torch
import ttnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--out", default="cb_allocator.json")
    a = ap.parse_args()

    from tt_bio import reblock_permute as rbp
    from tt_bio import tenstorrent as tt_dev

    device = tt_dev.get_device()
    rows = []
    try:
        x = ttnn.from_torch(torch.randn(1, a.n, a.n, a.c, dtype=torch.bfloat16),
                            dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        want = ttnn.to_torch(ttnn.permute(x, (0, 3, 1, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG))
        Ct = a.c // 32
        tile_bytes = 32 * 32 * 2
        for S in (1, 2, 4, 8):
            if Ct % S:
                continue
            for bufs in (1, 2):
                depth = 32 * S * bufs
                row = {"ct_stream": S, "bufs": bufs, "out_cb_tiles": depth,
                       "out_cb_kb": depth * tile_bytes / 1024,
                       "total_cb_kb": (depth + 4) * tile_bytes / 1024}
                rbp._CT_STREAM_PIN, rbp._CT_BUFS_PIN = str(S), str(bufs)
                rbp._CACHE.clear()
                try:
                    got = rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG)
                    ttnn.synchronize_device(device)
                    row["ran"] = True
                    row["bit_exact_vs_ttnn_permute"] = bool(torch.equal(ttnn.to_torch(got), want))
                    ttnn.deallocate(got)
                except Exception as e:  # allocator refusal or a kernel fault
                    row["ran"] = False
                    row["error"] = f"{type(e).__name__}: {e}"[:400]
                    traceback.print_exc()
                rows.append(row)
                print(json.dumps(row))
        # and the derived default, with no pin at all
        rbp._CT_STREAM_PIN = rbp._CT_BUFS_PIN = None
        rbp._CACHE.clear()
        S, depth = rbp._ct_stream(Ct, tile_bytes)
        got = rbp.reblock_permute(x, ttnn.DRAM_MEMORY_CONFIG)
        ttnn.synchronize_device(device)
        derived = {"ct_stream": S, "out_cb_tiles": depth, "out_cb_kb": depth * tile_bytes / 1024,
                   "bit_exact_vs_ttnn_permute": bool(torch.equal(ttnn.to_torch(got), want))}
        print(json.dumps({"derived_default": derived}))
        out = {"shape": [1, a.n, a.n, a.c], "ladder": rows, "derived_default": derived,
               "grid": [device.compute_with_storage_grid_size().x,
                        device.compute_with_storage_grid_size().y]}
        Path(a.out).write_text(json.dumps(out, indent=1))
    finally:
        tt_dev.cleanup()


if __name__ == "__main__":
    main()
