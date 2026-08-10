#!/usr/bin/env python3
"""T2 probe 4 — which kernel does the production channel move actually dispatch?

The row-major-detour hypothesis for `permute` @ 978/1065 rests so far on timing controls: an
untilize+tilize round trip costs about what the permute costs, and a whole-tile-only permute costs
the same as a sub-tile one. The build ships BOTH `*_permute_interleaved_rm_*` and
`*_permute_interleaved_tiled_*` kernels, so the hypothesis is directly observable: run the exact
production op against an EMPTY kernel cache and see which of the two gets compiled.

The cache root is `$HOME/.cache/tt-metal-cache`, so running with HOME pointed at a scratch
directory forces a cold compile of exactly the kernels this process uses, and nothing else.

Controls in the same process, so the snapshot is provably capturing what ran:
  - `ttnn.clone`  -> must compile `writer_unary_*` / `reader_unary_*`
  - `ttnn.transpose(-2,-1)` -> the tile-local op the DRAM path already uses

    HOME=/tmp/t2home python3 perf/t2_trimul/t2_probe4.py --out <json>
"""
import argparse
import json
import os
from pathlib import Path

import torch
import ttnn

from tt_bio.tenstorrent import get_device

L1 = ttnn.L1_MEMORY_CONFIG


def kernel_dirs():
    root = Path(os.path.expanduser("~/.cache/tt-metal-cache"))
    out = set()
    if root.exists():
        for d in root.glob("*/kernels/*"):
            if d.is_dir():
                out.add(d.name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = get_device()
    N, C = 320, 32
    res = {"home": os.path.expanduser("~"), "shape": [1, N, N, C]}

    base = kernel_dirs()
    res["after_device_open"] = sorted(base)
    print(f"kernels after device open: {len(base)}", flush=True)

    x = ttnn.from_torch(torch.randn(1, N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=L1)
    xb = ttnn.from_torch(torch.randn(1, C, N, N), layout=ttnn.TILE_LAYOUT, device=dev,
                         dtype=ttnn.bfloat16, memory_config=L1)

    steps = [
        ("clone_control", lambda: ttnn.clone(x, memory_config=L1)),
        ("transpose_m2_m1_control", lambda: ttnn.transpose(x, -2, -1, memory_config=L1)),
        ("permute_0312_PRODUCTION_in", lambda: ttnn.permute(x, (0, 3, 1, 2), memory_config=L1)),
        ("permute_0321_PRODUCTION_in_ending",
         lambda: ttnn.permute(x, (0, 3, 2, 1), memory_config=L1)),
        ("permute_0231_PRODUCTION_out", lambda: ttnn.permute(xb, (0, 2, 3, 1), memory_config=L1)),
    ]
    seen = set(base)
    for lbl, fn in steps:
        r = fn()
        ttnn.synchronize_device(dev)
        del r
        now = kernel_dirs()
        new = sorted(now - seen)
        seen = now
        res[lbl] = new
        print(f"  {lbl:34s} new kernels: {new}", flush=True)

    ttnn.deallocate(x)
    ttnn.deallocate(xb)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print("WROTE " + args.out, flush=True)


if __name__ == "__main__":
    main()
