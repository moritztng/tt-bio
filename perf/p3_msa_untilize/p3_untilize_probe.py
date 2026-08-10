#!/usr/bin/env python3
"""X3 deliverable 1: does the wide-last-dim untilize defect exist on qb1 card 2 at ttnn 0.67.4?

Roofs are re-measured on this card in this process (charter 4.1). Every timed region syncs on both
sides. Output is JSON on stdout plus a running log on stderr.
"""
from __future__ import annotations
import json, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN

DRAM, L1 = ttnn.DRAM_MEMORY_CONFIG, ttnn.L1_MEMORY_CONFIG
dev = get_device()
dg = dev.compute_with_storage_grid_size()
res = {"card": "qb1 card 2", "ttnn": getattr(ttnn, "__version__", "?"),
       "compute_grid": f"{dg.x}x{dg.y}", "core_grid_main": f"{CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y}"}
log = lambda *a: print(*a, file=sys.stderr, flush=True)
log(f"grid={dg.x}x{dg.y} core_grid_main={CORE_GRID_MAIN.x}x{CORE_GRID_MAIN.y} ttnn={res['ttnn']}")


def timed(fn, warm=2, pipe=3, reps=5):
    for _ in range(warm):
        fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        for _ in range(pipe):
            fn()
        ttnn.synchronize_device(dev)
        o.append((time.perf_counter() - t0) / pipe)
    return st.median(o) * 1e6          # us


# ---------------------------------------------------------------- roofs, this card, this process
log("=== roofs ===")
roofs = {}
for mb, n in ((16, 2048), (64, 4096), (128, 5792)):
    t = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    nbytes = n * n * 2
    us = timed(lambda: ttnn.deallocate(ttnn.clone(t, memory_config=DRAM)))
    gbs = 2 * nbytes / (us * 1e-6) / 1e9        # read + write
    roofs[f"dram_dram_copy_{nbytes//2**20}MB"] = {"us": round(us, 1), "GB/s": round(gbs, 1)}
    log(f"  DRAM->DRAM copy {nbytes//2**20:4d} MB  {us:9.1f} us  {gbs:7.1f} GB/s")
    try:
        us = timed(lambda: ttnn.deallocate(ttnn.clone(t, memory_config=L1)))
        roofs[f"dram_read_{nbytes//2**20}MB"] = {"us": round(us, 1),
                                                 "GB/s": round(nbytes / (us * 1e-6) / 1e9, 1)}
        log(f"  DRAM read (->L1)  {nbytes//2**20:4d} MB  {us:9.1f} us  "
            f"{nbytes/(us*1e-6)/1e9:7.1f} GB/s")
    except Exception as e:
        log(f"  DRAM read {nbytes//2**20} MB: {str(e)[:60]}")
    ttnn.deallocate(t)
# write roof: L1 -> DRAM at a size that fits L1
for n in (1024, 2048):
    try:
        t = ttnn.from_torch(torch.randn(n, n, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=L1)
    except Exception as e:
        log(f"  L1 alloc {n}x{n}: {str(e)[:60]}")
        continue
    nbytes = n * n * 2
    us = timed(lambda: ttnn.deallocate(ttnn.clone(t, memory_config=DRAM)))
    roofs[f"dram_write_{nbytes//2**20}MB"] = {"us": round(us, 1),
                                              "GB/s": round(nbytes / (us * 1e-6) / 1e9, 1)}
    log(f"  DRAM write (L1->) {nbytes//2**20:4d} MB  {us:9.1f} us  "
        f"{nbytes/(us*1e-6)/1e9:7.1f} GB/s")
    ttnn.deallocate(t)
res["roofs"] = roofs

# ------------------------------------------------------------- D1.1 / D1.4 the production shapes
log("=== D1.1 / D1.4: to_layout TILE->ROW_MAJOR at the fold's own shapes ===")
shapes = {}
for name, shp in (("(9536,9536) production", (9536, 9536)),
                  ("(298,1024,298) control", (298, 1024, 298)),
                  ("(298,32,9536) rank3 wide", (298, 32, 9536))):
    t = ttnn.from_torch(torch.zeros(*shp, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    nb = 1
    for d in shp:
        nb *= d
    nb *= 2
    us = timed(lambda: ttnn.deallocate(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)), warm=1, pipe=1, reps=3)
    clone = timed(lambda: ttnn.deallocate(ttnn.clone(t, memory_config=DRAM)), warm=1, pipe=1, reps=3)
    shapes[name] = {"MB": round(nb / 2**20, 1), "untilize_us": round(us, 1),
                    "untilize_GB/s": round(2 * nb / (us * 1e-6) / 1e9, 1),
                    "clone_us": round(clone, 1),
                    "clone_GB/s": round(2 * nb / (clone * 1e-6) / 1e9, 1)}
    log(f"  {name:26s} {nb/2**20:6.1f} MB  untilize {us:10.1f} us "
        f"({2*nb/(us*1e-6)/1e9:6.1f} GB/s)   clone {clone:8.1f} us "
        f"({2*nb/(clone*1e-6)/1e9:6.1f} GB/s)")
    ttnn.deallocate(t)
res["production_shapes"] = shapes

# ------------------------------------------------------ D1.3 width sweep at ~fixed total bytes
log("=== D1.3: last-dim width sweep, ~fixed total bytes (88804 tiles) ===")
TOT_TILES = 298 * 298
sweep = {}
for ct in (1, 2, 4, 8, 16, 32, 64, 128, 256, 298):
    rt = max(1, TOT_TILES // ct)
    rows, cols = rt * 32, ct * 32
    try:
        t = ttnn.from_torch(torch.zeros(rows, cols, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                            device=dev, dtype=ttnn.bfloat16, memory_config=DRAM)
    except Exception as e:
        log(f"  cols={cols:6d}: alloc {str(e)[:50]}")
        continue
    nb = rows * cols * 2
    us = timed(lambda: ttnn.deallocate(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)), warm=1, pipe=1, reps=3)
    sweep[cols] = {"rows": rows, "MB": round(nb / 2**20, 1), "us": round(us, 1),
                   "GB/s": round(2 * nb / (us * 1e-6) / 1e9, 1)}
    log(f"  cols={cols:6d} ({ct:3d} tiles)  rows={rows:7d}  {nb/2**20:6.1f} MB  "
        f"{us:10.1f} us  {2*nb/(us*1e-6)/1e9:7.1f} GB/s")
    ttnn.deallocate(t)
res["width_sweep"] = sweep

print(json.dumps(res, indent=1))
