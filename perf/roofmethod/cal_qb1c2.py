#!/usr/bin/env python3
"""Re-check T4's `engaged_cores` calibration points on qb1 card 2, with nothing else in the clock.

The first attempt put a `to_memory_config` reshard inside the timed callable, so it measured the
harness rather than the op and reported 16 of 110 for a full-grid unary. Here the tensors are
pre-sharded on the grid under test and the timed region contains only the op.
"""
import json
import sys

import torch
import ttnn

from tt_bio.tenstorrent import get_device
sys.path.insert(0, "perf/ledger_298")
import util_probe                                                            # noqa: E402

dev = get_device()
out = {}

ROWS, COLS = 1280, 1408


def make_unary(mc):
    a = ttnn.from_torch(torch.randn(ROWS, COLS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=mc)
    c = ttnn.from_torch(torch.zeros(ROWS, COLS), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                        device=dev, memory_config=mc)
    return lambda: ttnn.mul(a, 1.0001, memory_config=mc, output_tensor=c)


r = util_probe.engaged_cores(dev, make_unary, (ROWS, COLS))
out["unary_full_grid"] = r
print(f"unary mul {ROWS}x{COLS}: engaged {r['engaged']} of {r['max_grid_cores']}  "
      f"floor_limited={r['floor_limited']}", flush=True)
print("  " + json.dumps(r["times_us"]), flush=True)

json.dump(out, open(sys.argv[1], "w"), indent=1)
print("wrote " + sys.argv[1], flush=True)
