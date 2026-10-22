"""Readback bandwidth vs DRAM page size: does a row-major output with bigger pages come home faster than 1.8 GB/s?

rb9 put the 300M/600M page batch's from_device at 4.1 / 4.8 ms (1.8 GB/s) with one 1920 / 2304 B page per row. The read
path moves one page at a time, so this times from_device of the same bytes laid out as [rows / k, k * hidden] row-major
(page = k rows), uploaded directly, plus the device cost of re-paging the shipped [rows, hidden] tensor with ttnn.reshape.

    python3 perf/spd_m_esmc/rb10.py --out rb10.json
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import ttnn

from tt_bio.tenstorrent import get_device

SHAPES = {"300m": (3808, 960), "600m": (3808, 1152)}
KS = (1, 2, 4, 8, 16, 32, 56, 112, 238)


def timed(fn, dev, rounds):
    for _ in range(2):
        fn()
    ttnn.synchronize_device(dev)
    ms = []
    for _ in range(rounds):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ms.append((time.perf_counter() - t0) * 1000)
    return round(statistics.median(ms), 3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = get_device()
    out = {}
    for name, (rows, h) in SHAPES.items():
        ref = torch.randn(rows, h).to(torch.bfloat16)
        r = {"mb": round(ref.numel() * 2 / 1e6, 2)}
        base = ttnn.from_torch(ref, device=dev, layout=ttnn.ROW_MAJOR_LAYOUT, dtype=ttnn.bfloat16)
        for k in KS:
            key = f"k{k}_page{k * h * 2}"
            try:
                t = ttnn.from_torch(ref.reshape(rows // k, k * h), device=dev, layout=ttnn.ROW_MAJOR_LAYOUT,
                                    dtype=ttnn.bfloat16)
                r[key] = timed(lambda: ttnn.from_device(t), dev, args.rounds)
                rs = ttnn.reshape(base, (rows // k, k * h))
                r[key + "_reshape_dev"] = timed(
                    lambda: ttnn.deallocate(ttnn.reshape(base, (rows // k, k * h))), dev, args.rounds)
                r[key + "_via_reshape"] = timed(lambda: ttnn.from_device(rs), dev, args.rounds)
                r[key + "_exact"] = bool(torch.equal(ttnn.to_torch(ttnn.from_device(rs)).reshape(rows, h), ref))
                ttnn.deallocate(t)
                ttnn.deallocate(rs)
            except Exception as exc:  # noqa: BLE001 - a refused page size is a result
                r[key] = repr(exc)[:200]
            print(name, key, r.get(key), r.get(key + "_via_reshape"), r.get(key + "_reshape_dev"), flush=True)
        ttnn.deallocate(base)
        out[name] = r
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
