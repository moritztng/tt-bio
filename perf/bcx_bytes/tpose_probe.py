#!/usr/bin/env python3
"""bcx-bytes: the dV product of a taped attention, `attn^T @ dO`, with and without ttnn's transpose.

`autograd.matmul`'s VJP asks `bmm(attn, g, transpose_a=True)`, and ttnn serves `transpose_a` with
a PermuteDeviceOperation on the whole [rows, heads, q, k] attention tensor. On the frontier block
at n=256 that permute ran at 10-38 % of the copy roof (`psum_prof_arms.json`), and it is the one
op that gets slower per byte when triangle attention stops chunking. `ttnn.transpose(-2, -1)` is
the tile-level WH transpose that the census found at 96 % of the copy roof on [1,256,256,1024].

Per shape: the permute alone and the transpose alone (bits equal?), then the full product both ways
(bits equal?), min and median of synced reps, AICLK sampled from sysfs during them.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "perf" / "bcx_bytes"


def main():
    import ttnn
    from perf.bcx_stack.stack import Clock
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock()
    cfg = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                 fp32_dest_acc_en=True, packer_l1_acc=True)

    def up(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)

    def timed(fn, reps=15):
        fn()
        ttnn.synchronize_device(dev)
        ts, spans = [], []
        for _ in range(reps):
            t0 = time.time()
            y = fn()
            ttnn.synchronize_device(dev)
            t1 = time.time()
            ts.append(t1 - t0)
            spans.append((t0, t1))
            ttnn.deallocate(y)
        ts.sort()
        return {"min_ms": ts[0] * 1e3, "median_ms": ts[len(ts) // 2] * 1e3,
                "aiclk": clock.window(spans)}

    rows = []
    g = torch.Generator().manual_seed(0)
    for shape in ([81, 4, 256, 256], [13, 4, 256, 256], [256, 4, 256, 256], [128, 4, 128, 128]):
        a = up(torch.softmax(torch.randn(shape, generator=g), -1))
        do = up(torch.randn(shape[:-1] + [32], generator=g))
        perm = ttnn.permute(a, (0, 1, 3, 2))
        tr = ttnn.transpose(a, -2, -1)
        same_move = bool((ttnn.to_torch(perm) == ttnn.to_torch(tr)).all())
        y0 = ag.bmm(a, do, True, False, compute_kernel_config=cfg)
        y1 = ag.bmm(tr, do, False, False, compute_kernel_config=cfg)
        same_mm = bool((ttnn.to_torch(y0) == ttnn.to_torch(y1)).all())
        for t in (perm, tr, y0, y1):
            ttnn.deallocate(t)
        mb = 2 * a.volume() * 2 / 1e6
        r = {"shape": shape, "move_MB": mb, "bits_permute_eq_transpose": same_move,
             "bits_product_eq": same_mm,
             "permute": timed(lambda: ttnn.permute(a, (0, 1, 3, 2))),
             "transpose": timed(lambda: ttnn.transpose(a, -2, -1)),
             "bmm_transpose_a": timed(lambda: ag.bmm(a, do, True, False, compute_kernel_config=cfg)),
             "transpose_then_bmm": timed(lambda: _tb(ag, ttnn, a, do, cfg))}
        for k in ("permute", "transpose"):
            r[k]["GBps_at_min"] = mb / r[k]["min_ms"]
        print(json.dumps(r), flush=True)
        rows.append(r)
        ttnn.deallocate(a)
        ttnn.deallocate(do)
    clock.stop()
    (OUT / "tpose_probe.json").write_text(json.dumps({"rows": rows}, indent=1))


def _tb(ag, ttnn, a, do, cfg):
    t = ttnn.transpose(a, -2, -1)
    y = ag.bmm(t, do, False, False, compute_kernel_config=cfg)
    ttnn.deallocate(t)
    return y


if __name__ == "__main__":
    main()
