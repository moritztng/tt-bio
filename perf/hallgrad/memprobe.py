#!/usr/bin/env python3
"""Measured activation cost of a taped backward, and the clock it ran at.

Reports device DRAM allocated after the forward (the retained set the tape holds) and after
the backward, at Protenix trunk pair shapes, so the 27.58 GB / 34.23 GB budget can be
checked against a measurement instead of an estimate.
"""
import json
import os
import subprocess
import sys
import threading
import time

import torch

SHAPES = [
    # (label, tokens, c_z) -- the pair tensor is [1, tokens, tokens, c_z]
    ("pair-128aa", 128, 256),
    ("pair-256aa", 256, 256),
]


def sample_aiclk(stop, out):
    """Sample AICLK while the work runs. A number without a clock is not a measurement."""
    while not stop.is_set():
        try:
            r = subprocess.run(["/home/ttuser/.local/bin/tt-smi", "-s"],
                               capture_output=True, text=True, timeout=25)
            d = json.loads(r.stdout)
            for i, dev in enumerate(d.get("device_info", [])):
                clk = dev.get("telemetry", {}).get("aiclk")
                if clk is not None:
                    out.setdefault(i, []).append(int(clk))
        except Exception:
            pass
        time.sleep(2.0)


def main():
    import ttnn
    from tt_bio import tenstorrent as tt
    from tt_bio import autograd as ag

    device = tt.get_device()
    stop = threading.Event()
    clocks = {}
    t = threading.Thread(target=sample_aiclk, args=(stop, clocks), daemon=True)
    t.start()

    mv0 = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    banks, per_bank = mv0.num_banks, mv0.total_bytes_per_bank
    total = banks * per_bank
    print(f"# DRAM {total / 1e9:.2f} GB ({total / 2 ** 30:.2f} GiB) over {banks} banks, "
          f"read from the live card")

    def dram_mb():
        mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
        return mv.total_bytes_allocated_per_bank * mv.num_banks / 2 ** 20

    print(f"{'shape':<12} {'tokens':>7} {'fwd_MB':>10} {'tape_MB':>10} {'bwd_peak_MB':>12} "
          f"{'grad_MB':>10}")
    for label, n, c_z in SHAPES:
        base = dram_mb()
        z = ag.Tensor(ttnn.from_torch(torch.randn(1, n, n, c_z), dtype=ttnn.bfloat16,
                                      layout=ttnn.TILE_LAYOUT, device=device),
                      requires_grad=True)
        w1 = ag.Tensor(ttnn.from_torch(torch.randn(c_z, c_z) / 16.0, dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=device),
                       requires_grad=True)
        w2 = ag.Tensor(ttnn.from_torch(torch.randn(c_z, c_z) / 16.0, dtype=ttnn.bfloat16,
                                       layout=ttnn.TILE_LAYOUT, device=device),
                       requires_grad=True)
        after_inputs = dram_mb()
        h = ag.layer_norm(z)
        h = ag.linear(h, w1)
        out = ag.linear(h, w2)
        after_fwd = dram_mb()
        out.backward()
        after_bwd = dram_mb()
        print(f"{label:<12} {n:>7} {after_fwd - after_inputs:>10.1f} "
              f"{after_fwd - base:>10.1f} {after_bwd - base:>12.1f} "
              f"{after_bwd - after_fwd:>10.1f}")
        for obj in (z, w1, w2, h, out):
            obj.value = None
            obj.grad = None
        del z, w1, w2, h, out

    mv = ttnn.get_memory_view(device, ttnn.BufferType.DRAM)
    print(f"\nDRAM free now {mv.total_bytes_free_per_bank * mv.num_banks / 1e9:.2f} GB "
          f"of {total / 1e9:.2f} GB")
    stop.set()
    t.join(timeout=5)
    for i, s in sorted(clocks.items()):
        if s:
            print(f"AICLK card{i} during run: min {min(s)} max {max(s)} "
                  f"mean {sum(s) / len(s):.0f} MHz over {len(s)} samples")
    return 0


if __name__ == "__main__":
    sys.exit(main())
