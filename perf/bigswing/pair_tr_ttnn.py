#!/usr/bin/env python3
"""Exec step 3a: is there a pure-ttnn expression of the pair transpose that beats 4.1989 ms?

The op is ttnn.permute(x, (1,0,2)) on [512,512,256] bf16 DRAM->DRAM. S118 established the mechanism:
dim 0 is the untiled batch and dim 1 is the tile-row axis, so the swap is not a whole-tile move --
output tile (j,R,C) takes one 32-element row from each of 32 different input tiles. Anything that
makes ttnn move whole tiles instead should show up here before a kernel gets written.

Every candidate is torch.equal-checked against the baseline's own output; a candidate that is not
exact is reported as WRONG and its time is not compared. Same process, same input tensor, warm 3,
median of 7, every intermediate freed.
"""
import argparse, json, statistics, sys, time
from pathlib import Path

import torch
import ttnn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device  # noqa: E402

DRAM = ttnn.DRAM_MEMORY_CONFIG
L1 = ttnn.L1_MEMORY_CONFIG
WARM, REPS = 3, 7


def timed(fn, dev):
    for _ in range(WARM):
        r = fn(); ttnn.synchronize_device(dev); ttnn.deallocate(r)
    ts = []
    r = None
    for i in range(REPS):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        r = fn()
        ttnn.synchronize_device(dev)
        ts.append(time.perf_counter() - t0)
        if i < REPS - 1:
            ttnn.deallocate(r)
    return statistics.median(ts), ts, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--c", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    N, C = a.n, a.c
    dev = get_device()
    res = {"host": "qb2", "chip": 0, "ttnn": "0.68.0", "n": N, "c": C,
           "loadavg": open("/proc/loadavg").read().split()[:3],
           "warm": WARM, "reps": REPS, "arms": []}
    torch.manual_seed(0)
    x = ttnn.from_torch(torch.randn(N, N, C), layout=ttnn.TILE_LAYOUT, device=dev,
                        dtype=ttnn.bfloat16, memory_config=DRAM)
    mb = N * N * C * 2 / 1e6

    def base():
        return ttnn.permute(x, (1, 0, 2), memory_config=DRAM)

    def transpose01():
        return ttnn.transpose(x, 0, 1, memory_config=DRAM)

    def via_rowmajor():
        rm = ttnn.to_layout(x, ttnn.ROW_MAJOR_LAYOUT)
        p = ttnn.permute(rm, (1, 0, 2), memory_config=DRAM)
        ttnn.deallocate(rm)
        o = ttnn.to_layout(p, ttnn.TILE_LAYOUT)
        ttnn.deallocate(p)
        return o

    def split_i_4d():
        """[512,512,256] -> [16,32,512,256], permute (2,0,1,3), reshape back.

        [j,I,ir,c] = x[I*32+ir, j, c] = x[i,j,c], and folding (I,ir) back into one axis is y."""
        x4 = ttnn.reshape(x, (N // 32, 32, N, C))
        p = ttnn.permute(x4, (2, 0, 1, 3), memory_config=DRAM)
        o = ttnn.reshape(p, (N, N, C))
        return o

    def batch4d():
        """The same op expressed 4D with a leading 1 -- a different ttnn dispatch path."""
        x4 = ttnn.reshape(x, (1, N, N, C))
        p = ttnn.permute(x4, (0, 2, 1, 3), memory_config=DRAM)
        return ttnn.reshape(p, (N, N, C))

    def l1_out():
        """Same permute, L1 destination -- the fit test declines this at 512 aa, so price it."""
        return ttnn.permute(x, (1, 0, 2), memory_config=L1)

    def clone_roof():
        return ttnn.clone(x, memory_config=DRAM)

    cands = [("permute_whole", base), ("transpose_0_1", transpose01),
             ("via_row_major", via_rowmajor), ("split_i_4d", split_i_4d),
             ("batch_4d", batch4d), ("l1_out", l1_out), ("clone_roof", clone_roof)]

    ref = None
    for name, fn in cands:
        row = {"arm": name}
        try:
            ms, ts, out = timed(fn, dev)
            got = ttnn.to_torch(out)
            ttnn.deallocate(out)
            row["ms"] = 1e3 * ms
            row["reps_ms"] = [1e3 * t for t in ts]
            row["GBs_rw"] = 2 * mb / 1e3 / ms
            if name == "permute_whole":
                ref = got
                row["exact"] = True
            elif name == "clone_roof":
                row["exact"] = None
            else:
                row["exact"] = bool(ref is not None and got.shape == ref.shape
                                    and torch.equal(got, ref))
        except Exception as e:                                          # noqa: BLE001
            row["error"] = f"{type(e).__name__}: {str(e)[:220]}"
        res["arms"].append(row)
        print(json.dumps(row)[:300], flush=True)

    ttnn.deallocate(x)
    res["loadavg_end"] = open("/proc/loadavg").read().split()[:3]
    Path(a.out).write_text(json.dumps(res, indent=1))
    b = next(r for r in res["arms"] if r["arm"] == "permute_whole")
    print(f"\nbaseline {b['ms']:.4f} ms ({b['GBs_rw']:.1f} GB/s r+w), roof 352.5 GB/s slice-only")
    for r in res["arms"]:
        if r["arm"] in ("permute_whole", "clone_roof") or "ms" not in r:
            continue
        tag = "EXACT" if r.get("exact") else "WRONG"
        print(f"  {r['arm']:16s} {r['ms']:8.4f} ms  {b['ms']/r['ms']:6.4f}x  {tag}")


main()
