#!/usr/bin/env python3
"""Measured streaming roof on this part: the denominator for every achieved-bandwidth number.

A roof is measured, not asserted. Three shapes of pure movement -- a clone (1 read 1 write), an
in-place add (2 reads 1 write) and an eltwise multiply into a fresh tensor (2 reads 1 write) --
at the pair tensor's own size, plus the same at 4x it, so the fit is not read off one point.
"""
import argparse, json, statistics, time
from pathlib import Path
import torch, ttnn
from tt_bio import tenstorrent as TT


def bench(fn, iters, device):
    ts = []
    for _ in range(iters):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(device)
        ts.append(time.perf_counter() - t0)
        if isinstance(out, ttnn.Tensor):
            ttnn.deallocate(out)
    return statistics.median(ts), min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=9)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    dev = TT.get_device()
    g = dev.compute_with_storage_grid_size()
    rows = []
    for mult, shape in ((1, (1, 512, 512, 128)), (4, (1, 1024, 1024, 128))):
        n = 1
        for d in shape:
            n *= d
        B = n * 2
        x = ttnn.from_torch(torch.randn(*shape) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        y = ttnn.from_torch(torch.randn(*shape) * 0.05, layout=ttnn.TILE_LAYOUT, device=dev,
                            dtype=ttnn.bfloat16)
        for name, fn, nb in (
            ("clone", lambda: ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG), 2),
            ("multiply", lambda: ttnn.multiply(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG), 3),
            ("add", lambda: ttnn.add(x, y, memory_config=ttnn.DRAM_MEMORY_CONFIG), 3),
        ):
            fn()  # compile
            med, lo = bench(fn, a.iters, dev)
            rows.append({"shape": "x".join(map(str, shape)), "op": name, "moved_B": nb * B,
                         "median_ms": med * 1e3, "min_ms": lo * 1e3,
                         "median_GBps": nb * B / med / 1e9, "peak_GBps": nb * B / lo / 1e9})
        ttnn.deallocate(x); ttnn.deallocate(y)
    res = {"arch": str(dev.arch()), "grid": f"{g.x}x{g.y}", "n_cores": g.x * g.y,
           "l1_unreserved_per_core_B": ttnn.get_max_worker_l1_unreserved_size(),
           "rows": rows, "roof_GBps": max(r["peak_GBps"] for r in rows)}
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    for r in rows:
        print(f"{r['shape']:20s} {r['op']:9s} {r['moved_B']/1e6:8.1f} MB  "
              f"{r['median_ms']:7.3f} ms  {r['median_GBps']:7.1f} GB/s  peak {r['peak_GBps']:7.1f}")
    print(f"\nMEASURED STREAMING ROOF on {res['arch']} {res['grid']}: {res['roof_GBps']:.1f} GB/s")


if __name__ == "__main__":
    main()
