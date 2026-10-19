"""DRAM write ceiling on one chip, to judge triatt_ln's writer (553 MB a call at Protenix-v2's 512 x 512 x 256).

    TT_VISIBLE_DEVICES=<card> python perf/spd_m_protenix/wbw.py [--mb 553] [--reps 10]

Arms, each over a bf16 TILE DRAM-interleaved tensor of --mb megabytes: `full` (ttnn.full on device: writes only)
and `clone` (ttnn.clone: reads and writes the same bytes). Prints ms and GB/s (bytes written / time, and for clone
also bytes moved) per arm, median of --reps after 3 warm calls, one JSON line each.
"""
import argparse, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("--mb", type=float, default=553)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()

from tt_bio.main import ensure_p300_mesh_descriptor  # noqa: E402
ensure_p300_mesh_descriptor()
import torch  # noqa: E402
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402

dev = T.get_device()
rows = int(a.mb * 1e6 / (2 * 1024)) // 32 * 32
shape = [1, 1, rows, 1024]
nbytes = rows * 1024 * 2
src = ttnn.from_torch(torch.zeros(shape), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)


def arm(name, fn, moved):
    for _ in range(3):
        ttnn.deallocate(fn())
    ttnn.synchronize_device(dev)
    ts = []
    for _ in range(a.reps):
        t0 = time.perf_counter()
        o = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        ttnn.deallocate(o)
    ms = statistics.median(ts) * 1e3
    print(json.dumps({"arm": name, "mb": nbytes / 1e6, "ms": round(ms, 3), "write_gbs": round(nbytes / ms / 1e6, 1),
                      "moved_gbs": round(moved * nbytes / ms / 1e6, 1)}), flush=True)


arm("full", lambda: ttnn.full(shape, 1.0, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev), 1)
arm("clone", lambda: ttnn.clone(src, memory_config=ttnn.DRAM_MEMORY_CONFIG), 2)
