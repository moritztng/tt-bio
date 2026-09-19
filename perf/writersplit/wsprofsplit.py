#!/usr/bin/env python3
"""Falsifier (b): does BRISC's residency fall, and does the op fall with it?

One arm per process, selected by --flag, because the two arms are the same op code on the same
shape with the same output memory and the reducer separates arms by exactly those fields.  Run
each under `python3 -m tracy -p --enable-sum-profiling -o <dir>`; kt=4 and kt=32 fall out on
their own inside a run, they differ in INPUT_0_X_PAD.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "perf/writeside"))

import clk  # noqa: E402
import ttnn  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
B, M, N = 16, 512, 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--flag", default="0")
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--clock", type=int, default=1350)
    a = ap.parse_args()
    os.environ["TTNN_MM2D_WRITER_ON_IN0"] = a.flag

    import time

    import torch
    from tt_bio import tenstorrent as T

    device = T.get_device()
    held = clk.force(a.clock, clk.nodes_open_by_this_process())
    t0 = time.time()
    while time.time() - t0 < 10.0 and not all(clk.aiclk(n) >= a.clock - 5 for n in held):
        time.sleep(0.01)
    reached = {n: clk.aiclk(n) for n in held}
    if any(v < a.clock - 5 for v in reached.values()):
        print("REFUSING to measure: %r" % (reached,))
        return 2
    print("arm flag=%s, nodes %r forced to %d MHz" % (a.flag, held, a.clock), flush=True)
    sampler = clk.Sampler(held[0])

    GRID = T.CORE_GRID_MAIN
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)

    def dev(t):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=DRAM)

    shapes = {kt: (dev(torch.randn(1, B, M, 32 * kt) * 0.05), dev(torch.randn(32 * kt, N) * 0.05))
              for kt in (4, 32)}

    def call(kt):
        x, w = shapes[kt]
        return ttnn.linear(x, w, compute_kernel_config=KC, memory_config=DRAM,
                           dtype=ttnn.bfloat16, core_grid=GRID)

    for kt in (4, 32):
        for _ in range(3):
            r = call(kt)
            ttnn.synchronize_device(device)
            ttnn.deallocate(r)
        live = []
        for _ in range(a.calls):
            live.append(call(kt))
            if len(live) >= 8:
                ttnn.synchronize_device(device)
                for r in live:
                    ttnn.deallocate(r)
                live = []
        ttnn.synchronize_device(device)
        for r in live:
            ttnn.deallocate(r)
        print("arm flag=%s kt=%d: %d calls" % (a.flag, kt, a.calls), flush=True)

    print("AICLK during: %r" % (sampler.stop(),), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
