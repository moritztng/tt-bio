#!/usr/bin/env python3
"""The two tri_att pair transposes cost 4.432 s/fold and one of them is bfloat8_b.

The 512 aa `--fast` census times `ttnn.permute(x, (1,0,2))` on a [512,512,256] DRAM tensor at
4.069 ms in bfloat16 (tenstorrent.py:2032) and 5.165 ms in bfloat8_b (tenstorrent.py:2181). The
bfp8 arm moves 0.53x the bytes and takes 1.27x the time. This separates the transpose from the
move: a clone at the same shape and dtype is the same traffic with no layout change.

    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:pairformer-resident-chunking \
        python3 perf/bigswing/permute_dtype.py --out perf/bigswing/permute_dtype_qb2c0.json
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device  # noqa: E402

DT = {"bf16": ttnn.bfloat16, "bfp8": ttnn.bfloat8_b}
BYTES = {"bf16": 2.0, "bfp8": 1.0625}


def timed(fn, dev, warm=2, reps=5):
    for _ in range(warm):
        r = fn()
        del r
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
    res = {"n": a.n, "c": a.c, "loadavg": open("/proc/loadavg").read().split()[:3],
           "ttnn": getattr(ttnn, "__version__", "?"), "arms": {}}
    torch.manual_seed(0)
    host = torch.randn(a.n, a.n, a.c)

    for tag, dt in DT.items():
        x = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt,
                            memory_config=ttnn.DRAM_MEMORY_CONFIG)
        mb = a.n * a.n * a.c * BYTES[tag] / 1e6
        for name, fn in (
            ("permute", lambda x=x: ttnn.permute(x, (1, 0, 2),
                                                 memory_config=ttnn.DRAM_MEMORY_CONFIG)),
            ("clone", lambda x=x: ttnn.clone(x, memory_config=ttnn.DRAM_MEMORY_CONFIG)),
            ("transpose01", lambda x=x: ttnn.transpose(x, 0, 1,
                                                       memory_config=ttnn.DRAM_MEMORY_CONFIG)),
        ):
            try:
                med, all_ = timed(fn, dev)
                res["arms"][f"{name}.{tag}"] = {
                    "ms": med * 1e3, "all_ms": [t * 1e3 for t in all_], "MB": mb,
                    "GBps_rw": 2 * mb / 1e3 / med}
            except Exception as e:                                       # noqa: BLE001
                res["arms"][f"{name}.{tag}"] = {"error": f"{type(e).__name__}: {e}"[:200]}
                ttnn.synchronize_device(dev)
        ttnn.deallocate(x)

    # bfp8 -> bf16 -> permute -> bfp8: is casting around the transpose cheaper than doing it in bfp8
    xb = ttnn.from_torch(host, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat8_b,
                         memory_config=ttnn.DRAM_MEMORY_CONFIG)

    def cast_round():
        u = ttnn.typecast(xb, ttnn.bfloat16, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        p = ttnn.permute(u, (1, 0, 2), memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(u)
        d = ttnn.typecast(p, ttnn.bfloat8_b, memory_config=ttnn.DRAM_MEMORY_CONFIG)
        ttnn.deallocate(p)
        return d

    try:
        med, all_ = timed(cast_round, dev)
        res["arms"]["cast_permute_cast.bfp8"] = {"ms": med * 1e3,
                                                 "all_ms": [t * 1e3 for t in all_]}
    except Exception as e:                                               # noqa: BLE001
        res["arms"]["cast_permute_cast.bfp8"] = {"error": f"{type(e).__name__}: {e}"[:200]}

    json.dump(res, open(a.out, "w"), indent=1)
    print(f"n={a.n} c={a.c} load={res['loadavg']}")
    for k, v in res["arms"].items():
        if "error" in v:
            print(f"  {k:<26} ERROR {v['error']}")
        else:
            extra = f"{v['MB']:8.1f} MB  {v['GBps_rw']:7.1f} GB/s r+w" if "MB" in v else ""
            print(f"  {k:<26} {v['ms']:8.4f} ms  {extra}")


if __name__ == "__main__":
    main()
