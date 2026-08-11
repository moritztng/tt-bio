#!/usr/bin/env python3
"""Why the pair transpose runs at 18 % (bf16) / 7 % (bfp8) of the clone rate at the same shape.

permute_dtype_512_qb2c0.json: clone 0.756 ms, permute 4.235 ms on [512,512,256] bf16 DRAM.
Two candidate mechanisms:
  U -- ttnn untilizes, permutes row-major, retilizes. Then to_layout alone accounts for most of it
       and the bfp8 penalty is the block-float pack/unpack, not the transaction size.
  T -- ttnn moves whole tiles but one transaction per tile (2048 B bf16, 1088 B bfp8), so the rate
       tracks transaction size. Then row-blocking, which makes each transfer a contiguous run of
       tile-rows, recovers it.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:pairformer-resident-chunking \
        python3 perf/bigswing/permute_mech.py --out perf/bigswing/permute_mech_512_qb2c0.json
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG


def timed(fn, dev, warm=2, reps=5):
    for _ in range(warm):
        r = fn(); del r
    ttnn.synchronize_device(dev)
    out = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        out.append(time.perf_counter() - t0)
        del r
    return statistics.median(out), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = get_device()
    res = {"n": a.n, "c": a.c, "loadavg": open("/proc/loadavg").read().split()[:3], "arms": {}}
    torch.manual_seed(0)
    host = torch.randn(a.n, a.n, a.c)
    x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                        memory_config=DRAM)
    mb = a.n * a.n * a.c * 2 / 1e6

    def rowblock(R):
        def go():
            parts = [ttnn.permute(x[i:i + R], (1, 0, 2), memory_config=DRAM)
                     for i in range(0, a.n, R)]
            out = ttnn.concat(parts, dim=1, memory_config=DRAM)
            for p in parts:
                ttnn.deallocate(p)
            return out
        return go

    arms = {
        "permute": lambda: ttnn.permute(x, (1, 0, 2), memory_config=DRAM),
        "clone": lambda: ttnn.clone(x, memory_config=DRAM),
        "to_row_major": lambda: ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT),
        "slice_only_R64": lambda: [x[i:i + 64] for i in range(0, a.n, 64)],
    }
    for R in (32, 64, 128, 256):
        arms[f"rowblock_R{R}"] = rowblock(R)

    for name, fn in arms.items():
        try:
            med, all_ = timed(fn, dev)
            res["arms"][name] = {"ms": med * 1e3, "all_ms": [t * 1e3 for t in all_],
                                 "GBps_rw": 2 * mb / 1e3 / med}
        except Exception as e:                                           # noqa: BLE001
            res["arms"][name] = {"error": f"{type(e).__name__}: {e}"[:220]}
            ttnn.synchronize_device(dev)

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"n={a.n} c={a.c} bf16 {mb:.1f} MB  load={res['loadavg']}")
    for k, v in res["arms"].items():
        print(f"  {k:<20} " + (f"ERROR {v['error']}" if "error" in v
                               else f"{v['ms']:8.4f} ms  {v['GBps_rw']:7.1f} GB/s r+w"))


if __name__ == "__main__":
    main()
