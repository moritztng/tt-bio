"""Is the trimul gated backward bound by DRAM or by its SFPU math? (bc4-bet-block, K1 go/no-go)

A fused trimul in-projection VJP (gather da, gate backward, x W^T in L1) removes the gated backward's
DRAM writes but not its sigmoid/multiply work. Same reader, three kernels at N tokens, C=128:

  move   reblock_permute_back(da)                       reads 1P, writes 1P, no math
  gated  reblock_permute_gated_bw(da, xw, 0, 2C)        reads 3P, writes 2P, sigmoid + 5 mul in fp32 DST
  clone  ttnn.clone(xw) of [1,N,N,4C]                   reads 4P, writes 4P: the card's achieved copy rate

gated's time against (its bytes / clone's rate) is its DRAM-bound time; the excess is math the fusion keeps.

    TT_VISIBLE_DEVICES=<c> TT_BIO_LEASE_CARDS=<c> TT_BIO_LEASE_HOLDER=worker:bc4-bet-block \
        timeout 900 python3 perf/bc4_block/gated_bw_bound.py OUT.json [N ...]
"""
import json
import sys
import time

import torch
import ttnn

import tt_bio.reblock_permute as R
import tt_bio.tenstorrent as T

dev = T.get_device()
C, REPS = 128, 30


def up(*shape):
    return ttnn.from_torch(torch.randn(*shape).bfloat16(), layout=ttnn.TILE_LAYOUT, device=dev,
                           dtype=ttnn.bfloat16)


def timed(fn):
    for _ in range(3):
        fn()
    ttnn.synchronize_device(dev)
    t = time.perf_counter()
    for _ in range(REPS):
        fn()
    ttnn.synchronize_device(dev)
    return (time.perf_counter() - t) / REPS


def run(n):
    P = n * n * C * 2
    da, xw = up(1, C, n, n), up(1, n, n, 4 * C)
    out = {}

    def move():
        ttnn.deallocate(R.reblock_permute_back(da))

    def gated():
        dp, dg = R.reblock_permute_gated_bw(da, xw, 0, 2 * C)
        ttnn.deallocate(dp)
        ttnn.deallocate(dg)

    def clone():
        ttnn.deallocate(ttnn.clone(xw))

    for name, fn, nbytes in (("move", move, 2 * P), ("gated", gated, 5 * P), ("clone", clone, 8 * P)):
        s = timed(fn)
        out[name] = {"ms": s * 1e3, "GB_s": nbytes / s / 1e9, "bytes": nbytes}
    rate = out["clone"]["GB_s"]
    out["gated_dram_bound_ms"] = 5 * P / rate / 1e6
    out["gated_excess_ms"] = out["gated"]["ms"] - out["gated_dram_bound_ms"]
    out["gated_stats"] = list(R.STATS_GATED_BW)
    return out


if __name__ == "__main__":
    ns = [int(x) for x in sys.argv[2:]] or [224, 288]
    res = {"n": {}, "reps": REPS}
    try:
        import subprocess
        res["sha"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    for n in ns:
        res["n"][n] = run(n)
        print(n, json.dumps(res["n"][n]), flush=True)
    json.dump(res, open(sys.argv[1], "w"), indent=1)
