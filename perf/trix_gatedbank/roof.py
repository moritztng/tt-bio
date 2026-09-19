#!/usr/bin/env python3
"""Where the two channel-move legs sit against the measured Blackhole DRAM roof.

Both are timed at the shape `firing_512_qb1c0.json` counted them at, in batches between two
synchronize_device calls, arms interleaved. The byte count is the op's own traffic: the gated move
reads two 128-channel slices of the wide projection and writes one, the back move reads and writes
one each. 435.2 GB/s is the fleet's measured p150a DRAM roof, not a datasheet number.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

N, CW, SLICE_C = 512, 512, 128
P_SLICE, G_SLICE = 8 * 32, 12 * 32
DRAM_ROOF_GBS = 435.2


def clocks():
    return {str(n): int(Path(f"/sys/class/tenstorrent/tenstorrent!{n}/tt_aiclk").read_text())
            for n in range(4)}


def main():
    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio import reblock_permute as rbp

    dev = get_device()
    mc = ttnn.DRAM_MEMORY_CONFIG
    torch.manual_seed(0)
    xw = ttnn.from_torch((torch.randn(1, N, N, CW) * 2).to(torch.bfloat16),
                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    xb = ttnn.from_torch((torch.randn(1, SLICE_C, N, N) * 2).to(torch.bfloat16),
                         dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev, memory_config=mc)
    og = ttnn.allocate_tensor_on_device(
        ttnn.Shape([1, SLICE_C, N, N]), ttnn.bfloat16, ttnn.TILE_LAYOUT, dev, mc)

    el = 2
    bytes_gated = (2 * N * N * SLICE_C + SLICE_C * N * N) * el
    bytes_back = (SLICE_C * N * N + N * N * SLICE_C) * el

    def gated(n):
        for _ in range(n):
            rbp.reblock_permute_gated(xw, P_SLICE, G_SLICE, SLICE_C, memory_config=mc,
                                      device=dev, out=og, row_off=0)

    keep = []

    def back(n):
        keep.clear()
        for _ in range(n):
            o = rbp.reblock_permute_back(xb, memory_config=mc, device=dev)
            ttnn.deallocate(o)

    n, rounds = 40, 9
    gated(3); back(3)
    ttnn.synchronize_device(dev)
    samples = {"gated": [], "back": []}
    clk = []
    for _ in range(rounds):
        for nm, fn in (("gated", gated), ("back", back)):
            ttnn.synchronize_device(dev)
            t0 = time.perf_counter()
            fn(n)
            ttnn.synchronize_device(dev)
            samples[nm].append((time.perf_counter() - t0) / n * 1e3)
            clk.append(clocks())

    med = {k: statistics.median(v) for k, v in samples.items()}
    calls = {"gated": 1120, "back": 560}
    out = {
        "host": os.uname().nodename, "batch": n, "rounds": rounds,
        "fold_s_512aa_measured": 27.714,
        "bytes_per_call": {"gated": bytes_gated, "back": bytes_back},
        "ms_per_call_median": {k: round(v, 5) for k, v in med.items()},
        "ms_per_call_all": {k: [round(x, 5) for x in v] for k, v in samples.items()},
        "gbs": {k: round(b / (med[k] * 1e-3) / 1e9, 1)
                for k, b in (("gated", bytes_gated), ("back", bytes_back))},
        "pct_of_dram_roof": {k: round(b / (med[k] * 1e-3) / 1e9 / DRAM_ROOF_GBS * 100, 1)
                             for k, b in (("gated", bytes_gated), ("back", bytes_back))},
        "calls_per_fold": calls,
        "s_per_fold": {k: round(med[k] * calls[k] / 1e3, 4) for k in med},
        "share_of_fold_pct": {k: round(med[k] * calls[k] / 1e3 / 27.714 * 100, 2) for k in med},
        "dram_roof_gbs": DRAM_ROOF_GBS,
        "aiclk_during": {str(i): sorted(set(c[str(i)] for c in clk)) for i in range(4)},
        "loadavg": [round(v, 2) for v in os.getloadavg()],
    }
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
