#!/usr/bin/env python3
"""bcx-reduce: the host cost of `taped_ttnn._permute_back`, reblock kernel against `ttnn.permute`.

The device profile credits the reblock backward 5.9 ms per block, and the whole step read slower
with it on. The step is host-bound, so this reads what each path costs the host thread: the time
to ENQUEUE a burst of calls with no sync inside it (the host's own cost; the device runs behind
it), and the synced per-call wall beside it. Arms interleave burst by burst in one process.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "perf" / "bcx_reduce"


def main():
    import ttnn
    from perf.bcx_stack.stack import Clock, sysfs_node
    from tt_bio import taped_ttnn as T
    from tt_bio.tenstorrent import get_device
    dev = get_device()
    clock = Clock(dt=0.02)
    burst, bursts = 24, 12
    blob = {"pci": sysfs_node()[1], "loadavg_start": os.getloadavg(), "burst": burst,
            "bursts": bursts, "shapes": []}
    for shape, inv in (([1, 64, 256, 256], [0, 2, 3, 1]), ([1, 256, 256, 64], [0, 3, 1, 2])):
        g = ttnn.from_torch(torch.randn(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT,
                            device=dev)
        arms = {"permute": lambda: ttnn.permute(g, inv), "reblock": lambda: T._permute_back(g, inv)}
        T.REBLOCK_PERMUTE_BW = True
        same = torch.equal(ttnn.to_torch(T._permute_back(g, inv)), ttnn.to_torch(ttnn.permute(g, inv)))
        enq, wall = {a: [] for a in arms}, {a: [] for a in arms}
        spans = []
        for fn in arms.values():                    # warm: compile + cache the descriptor
            for _ in range(3):
                ttnn.deallocate(fn())
        ttnn.synchronize_device(dev)
        for b in range(bursts):
            for a in (list(arms) if b % 2 == 0 else list(arms)[::-1]):
                fn = arms[a]
                ttnn.synchronize_device(dev)
                ta = time.time()
                t0 = time.perf_counter()
                ys = [fn() for _ in range(burst)]
                t1 = time.perf_counter()
                ttnn.synchronize_device(dev)
                t2 = time.perf_counter()
                spans.append((ta, time.time()))
                enq[a].append((t1 - t0) / burst * 1e3)
                wall[a].append((t2 - t0) / burst * 1e3)
                for y in ys:
                    ttnn.deallocate(y)
        row = {"shape": shape, "inv": inv, "bit_identical_to_permute": bool(same),
               "aiclk": clock.window(spans), "loadavg": os.getloadavg()}
        for a in arms:
            row[a] = {"enqueue_ms_per_call_median": float(np.median(enq[a])),
                      "enqueue_ms_per_call_min": float(min(enq[a])),
                      "wall_ms_per_call_median": float(np.median(wall[a])),
                      "wall_ms_per_call_min": float(min(wall[a]))}
        print(json.dumps(row), flush=True)
        blob["shapes"].append(row)
    clock.stop()
    (OUT / "host.json").write_text(json.dumps(blob, indent=1))


if __name__ == "__main__":
    main()
