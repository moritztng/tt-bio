#!/usr/bin/env python3
"""Bit-exactness across the shapes the relocation's padding maths can break.

The op-level falsifier runs one clean shape. This one walks the ragged cases: M and N not
multiples of the per-core block, several batch counts, a grid that leaves an H-dim tail, and a
1-core-tall case where the mcast sender core is itself the last block row. Each shape is run on
both arms in one process and compared with torch.equal.
"""
from __future__ import annotations

import os
import sys

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG

SHAPES = [
    # (B, M, K, N, grid_y, grid_x)
    (1, 512, 128, 512, 10, 11),
    (16, 512, 128, 512, 10, 11),
    (1, 96, 64, 352, 3, 11),     # M/N not multiples of the block
    (1, 32, 32, 32, 1, 1),       # single core, sender core is the whole output
    (3, 160, 96, 288, 2, 7),     # batch > 1 with a ragged grid
    (1, 1024, 256, 64, 10, 11),  # tall and thin
    (2, 64, 512, 1024, 4, 8),
]


def main() -> int:
    device = ttnn.open_device(device_id=0)
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)
    bad = 0
    for (B, M, K, N, gy, gx) in SHAPES:
        grid = ttnn.CoreGrid(y=gy, x=gx)
        xt, wt = torch.randn(1, B, M, K) * 0.05, torch.randn(K, N) * 0.05

        def dev(t):
            return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                                   device=device, memory_config=DRAM)

        x_d, w_d = dev(xt), dev(wt)

        def arm(flag):
            os.environ["TTNN_MM_WRITER_ON_IN0"] = flag
            device.disable_and_clear_program_cache()
            device.enable_program_cache()
            outs = []
            for _ in range(2):
                r = ttnn.linear(x_d, w_d, compute_kernel_config=KC, memory_config=DRAM,
                                dtype=ttnn.bfloat16, core_grid=grid)
                ttnn.synchronize_device(device)
                outs.append(ttnn.to_torch(r))
                ttnn.deallocate(r)
            return outs

        try:
            ship, split = arm("0"), arm("1")
        except Exception as e:  # noqa: BLE001
            print("B=%-3d M=%-5d K=%-4d N=%-5d %2dx%-2d  RAISED %s" % (B, M, K, N, gy, gx, e), flush=True)
            bad += 1
            continue
        eq = all(torch.equal(a, b) for a, b in zip(ship, split))
        selfeq = torch.equal(split[0], split[1])
        d = max((a.float() - b.float()).abs().max().item() for a, b in zip(ship, split))
        print("B=%-3d M=%-5d K=%-4d N=%-5d %2dx%-2d  equal %-5s  split self %-5s  maxdiff %.3e"
              % (B, M, K, N, gy, gx, eq, selfeq, d), flush=True)
        bad += 0 if (eq and selfeq) else 1
        ttnn.deallocate(x_d)
        ttnn.deallocate(w_d)
    ttnn.close_device(device)
    print("VERDICT: %d/%d shapes bit-exact" % (len(SHAPES) - bad, len(SHAPES)), flush=True)
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
