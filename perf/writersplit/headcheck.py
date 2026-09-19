#!/usr/bin/env python3
"""Bit-exactness falsifier for the writer split, re-run on the tt-metal HEAD build.

Both arms run in one process against the same inputs on the same card. The program cache is
cleared between them so the factory re-reads TTNN_MM_WRITER_ON_IN0 and builds the other variant;
timing here is meaningless, correctness is not. Three calls per arm with a fresh output buffer
each time, because a relocated writer whose output address is not patched on a program-cache hit
would pass the first call and write to a stale buffer on the second.

Self-contained: imports ttnn out of the source build (PYTHONPATH), not tt_bio.
"""
from __future__ import annotations

import os
import sys

import torch
import ttnn

DRAM = ttnn.DRAM_MEMORY_CONFIG
B, M, K, N = 16, 512, 128, 512
GRID = ttnn.CoreGrid(y=10, x=11)


def main() -> int:
    device = ttnn.open_device(device_id=0)
    print("arch %r grid %r" % (device.arch(), GRID), flush=True)
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    torch.manual_seed(0)
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
        for _ in range(3):
            r = ttnn.linear(x_d, w_d, compute_kernel_config=KC, memory_config=DRAM,
                            dtype=ttnn.bfloat16, core_grid=GRID)
            ttnn.synchronize_device(device)
            outs.append(ttnn.to_torch(r))
            ttnn.deallocate(r)
        return outs

    ship = arm("0")
    split = arm("1")

    ok = all(torch.equal(ship[0], s) for s in ship[1:])
    print("shipped self-consistent over 3 calls: %s" % ok, flush=True)
    alleq = True
    for i, (a, b) in enumerate(zip(ship, split)):
        eq = torch.equal(a, b)
        alleq = alleq and eq
        d = (a.float() - b.float()).abs().max().item()
        print("call %d  torch.equal %-5s  max abs diff %.3e" % (i, eq, d), flush=True)

    ref = (xt.to(torch.float32) @ wt.to(torch.float32)).squeeze(0)
    for name, o in (("ship", ship[0]), ("split", split[0])):
        f = o.float().reshape(ref.shape)
        pcc = torch.corrcoef(torch.stack([f.flatten(), ref.flatten()]))[0, 1].item()
        print("%-6s vs fp32 torch: PCC %.6f" % (name, pcc), flush=True)
    ttnn.close_device(device)
    print("VERDICT: %s" % ("BIT-EXACT 3/3" if alleq and ok else "NOT BIT-EXACT"), flush=True)
    return 0 if (alleq and ok) else 1


if __name__ == "__main__":
    sys.exit(main())
