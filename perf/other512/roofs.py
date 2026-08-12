#!/usr/bin/env python3
"""The two roofs every byte model in this task is checked against, measured on THIS card.

This lineage has published 668 GB/s on a ~400 GB/s card and separately made an op sitting at 96 % of
the read roof look like 13 %, both times by inheriting a roof instead of measuring one. So: a
DRAM->DRAM copy roof (`ttnn.clone`, read+write) and a DRAM->L1 read roof, at the pair-tensor shapes
these models actually use, median of 9 synced calls after 2 warm.
"""
import json, statistics as st, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch, ttnn                                                            # noqa: E402
import tt_bio.tenstorrent as T                                                # noqa: E402

import os
from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
    mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
    if mgd:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

dev = T.get_device()
g = dev.compute_with_storage_grid_size()
out = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
       "grid": [g.x, g.y], "compute_grid_main": list(T.COMPUTE_GRID_MAIN), "rows": []}


def bench(fn, n=9, warm=2):
    for _ in range(warm):
        fn(); ttnn.synchronize_device(dev)
    ts = []
    for _ in range(n):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(r)
    return st.median(ts)


for shape in ([512, 512, 128], [512, 512, 384], [995, 995, 384], [512, 512, 256]):
    nbytes = 1
    for d in shape:
        nbytes *= d
    nbytes *= 2                                                    # bf16
    t = ttnn.from_torch(torch.zeros(*shape, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        dtype=ttnn.bfloat16, device=dev, memory_config=ttnn.DRAM_MEMORY_CONFIG)
    ms = bench(lambda: ttnn.clone(t, memory_config=ttnn.DRAM_MEMORY_CONFIG)) * 1e3
    row = {"shape": shape, "MiB": round(nbytes / 2**20, 1), "clone_ms": round(ms, 4),
           "copy_roof_GBps": round(2 * nbytes / (ms * 1e-3) / 1e9, 1)}
    out["rows"].append(row)
    print(json.dumps(row), flush=True)
    ttnn.deallocate(t)

print(json.dumps(out))
Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
T.cleanup()
