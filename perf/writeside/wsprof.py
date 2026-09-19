#!/usr/bin/env python3
"""Measure the writer's stall DIRECTLY, on the kernel, instead of inferring it from a roof.

`measure-math-thread-cb-stalls-directly-not-via-roofs` is the rule this row is built on: when an op
sits under a roof, measure the stall, do not name a third roof.  The two shipped sum accumulators
(`CB WAIT FRONT` on the unpack thread TRISC0, `CB RESERVE BACK` on the pack thread TRISC2) are the
only ones tt-metal ships, and `ttnn-stall-accumulator-cannot-discriminate-producer-vs-consumer`
warns they are compute-cluster-only -- blind to WHY a reader is stuck.  For the WRITE side that
limitation does not bite: `CB RESERVE BACK` is the pack thread blocked on room in the output CB,
and the only thing that frees that room is the writer draining it to DRAM.  It is exactly the
"is the write backpressuring the pipe" question, measured on the consumer side of the writer.

Read alongside it: the writer RISC's own kernel duration against the op's duration.  A writer that
is resident for the whole op and a pack thread that is blocked waiting for it is an unhidden write.
A writer resident for a fraction with no pack stall is a hidden one.

Arms, all at the fold's own key A (b=16, M=512, K=128, N=512) unless stated, separated in the CSV
by OUTPUT_0_MEMORY and INPUT_0_X_PAD so no call-index bookkeeping is needed:
  ship        output to DRAM        the shipped path
  out_l1      output to L1          the same op with the DRAM write removed
  k32 ship    K=1024, output DRAM   where the campaign's numbers say the term should shrink
  clone       L1 -> DRAM, 48 MiB    KNOWN-ANSWER CONTROL: a pure write, nothing else.  If the
                                    instrument is trustworthy this arm must read the writer
                                    pinned and the pack side irrelevant; the matmul arms are only
                                    readable against it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import clk                                                                     # noqa: E402
import ttnn                                                                    # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
B, M, K, N = 16, 512, 128, 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=40)
    ap.add_argument("--clock", type=int, default=1350)
    a = ap.parse_args()

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
    print("nodes %r forced to %d MHz" % (held, a.clock), flush=True)
    sampler = clk.Sampler(held[0])

    GRID = T.CORE_GRID_MAIN
    KC = ttnn.types.BlackholeComputeKernelConfig(
        math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    torch.manual_seed(0)

    def dev(t, mc=DRAM):
        return ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                               device=device, memory_config=mc)

    x_d, w_d = dev(torch.randn(1, B, M, K) * 0.05), dev(torch.randn(K, N) * 0.05)
    x32, w32 = dev(torch.randn(1, B, M, 1024) * 0.05), dev(torch.randn(1024, N) * 0.05)
    c_l1 = dev(torch.randn(3072, 4096) * 0.05, L1)

    def lin(xa, wa, mc):
        return lambda: ttnn.linear(xa, wa, compute_kernel_config=KC, memory_config=mc,
                                   dtype=ttnn.bfloat16, core_grid=GRID)

    arms = [("ship", lin(x_d, w_d, DRAM)),
            ("out_l1", lin(x_d, w_d, L1)),
            ("k32_ship", lin(x32, w32, DRAM)),
            ("clone_write", lambda: ttnn.clone(c_l1, memory_config=DRAM))]

    for name, fn in arms:
        for _ in range(3):                                    # warm: JIT and program cache
            r = fn()
            ttnn.synchronize_device(device)
            ttnn.deallocate(r)
        n = a.calls if name != "out_l1" else min(a.calls, 8)  # 8.39 MB x 40 does not fit L1
        live = []
        for i in range(a.calls):
            live.append(fn())
            if len(live) >= n:
                ttnn.synchronize_device(device)
                for r in live:
                    ttnn.deallocate(r)
                live = []
        ttnn.synchronize_device(device)
        for r in live:
            ttnn.deallocate(r)
        print("arm %-12s %d calls" % (name, a.calls), flush=True)

    print("AICLK during: %r" % (sampler.stop(),), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
