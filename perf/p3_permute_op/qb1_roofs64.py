#!/usr/bin/env python3
"""The copy floors at the shape the fold actually runs, [1, 298, 298, 64], on qb1 card 0.

The earlier probe measured its floors at C=32, which is half this tensor's bytes, so the denominator
for every ratio at the production shape was missing. Measured here in BOTH instruments, because on
this op they disagree in sign: `synced` charges host and per-call dispatch serially, `thru` enqueues
K calls back to back and syncs once, which is what a fold does.

    TT_VISIBLE_DEVICES=0 python3 perf/p3_permute_op/qb1_roofs64.py
"""
from __future__ import annotations

import argparse, json, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import torch, ttnn

MC = {"l1": ttnn.L1_MEMORY_CONFIG, "dram": ttnn.DRAM_MEMORY_CONFIG}


def timeit(device, fn, reps=21, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    ts = []
    for _ in range(reps):
        ttnn.synchronize_device(device)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(device)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return ts[len(ts) // 2]


def thru(device, fn, k=40, warmup=5):
    for _ in range(warmup):
        fn()
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(k):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) * 1e6 / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "qb1_roofs64.json"))
    a = ap.parse_args()

    from tt_bio import reblock_permute as RP
    import tt_bio.tenstorrent as T
    RP.set_enabled(True)
    device = T.get_device()
    g = device.compute_with_storage_grid_size()
    R = {"wheel": "0.67.4", "host": "qb1", "card": 0, "device_grid": [int(g.x), int(g.y)],
         "shape": [1, 298, 298, 64], "bytes_one_way": 298 * 298 * 64 * 2, "rows": []}

    ref = torch.randn(1, 298, 298, 64, dtype=torch.bfloat16)
    nb = 298 * 298 * 64 * 2
    for si in ("l1", "dram"):
        x = ttnn.from_torch(ref, layout=ttnn.TILE_LAYOUT, dtype=ttnn.bfloat16, device=device,
                            memory_config=MC[si])
        for so in ("l1", "dram"):
            f = lambda x=x, so=so: ttnn.deallocate(ttnn.clone(x, memory_config=MC[so]))
            row = {"op": "clone", "src": si, "dst": so,
                   "synced_us": round(timeit(device, f), 2), "thru_us": round(thru(device, f), 2)}
            row["thru_GBs_one_way"] = round(nb / row["thru_us"] / 1e3, 1)
            R["rows"].append(row); print(row, flush=True)
        # the op and the stock call it replaces, same source, destination = source
        for name, f in (("reblock_permute",
                         lambda x=x, si=si: ttnn.deallocate(RP.reblock_permute(x, MC[si], device))),
                        ("ttnn.permute(0,3,1,2)",
                         lambda x=x, si=si: ttnn.deallocate(
                             ttnn.permute(x, (0, 3, 1, 2), memory_config=MC[si])))):
            f()
            row = {"op": name, "src": si, "dst": si,
                   "synced_us": round(timeit(device, f), 2), "thru_us": round(thru(device, f), 2)}
            row["thru_GBs_one_way"] = round(nb / row["thru_us"] / 1e3, 1)
            R["rows"].append(row); print(row, flush=True)
        ttnn.deallocate(x)

    # cores engaged at this shape, read out of the work split
    ent = next(iter(RP._CACHE.values()))
    n = 0
    for cr in ent["core_grid"].ranges():
        n += (cr.end.x - cr.start.x + 1) * (cr.end.y - cr.start.y + 1)
    R["cores_engaged"] = n
    R["cores_on_grid"] = int(g.x) * int(g.y)
    print("cores", n, "of", R["cores_on_grid"], flush=True)

    Path(a.out).write_text(json.dumps(R, indent=2) + "\n")
    T.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
