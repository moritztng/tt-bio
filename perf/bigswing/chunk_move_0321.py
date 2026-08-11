#!/usr/bin/env python3
"""tenstorrent.py:1709 -- the (0,3,2,1) chunk move at 512 aa takes the branch written for small L.

`_transform_chunk` decomposes (0,3,2,1) into [channel_move (0,3,1,2), transpose(-2,-1)] only when
`memory_config.buffer_type == DRAM`, on the reasoning that "on the small-L L1 path the single
permute is marginally faster". At 512 aa the chunk config is L1 and L=512, the largest in the
campaign, so the guard keys on buffer type where the comment's reasoning keys on L. 16 calls per
block, 1.534 s/fold. This times the three ways of doing it in one process.
"""
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path
import torch, ttnn
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device            # noqa: E402
import tt_bio.reblock_permute as rp                  # noqa: E402
WARM, REPS = 3, 7

def med(fn, dev):
    ts = []
    for i in range(WARM + REPS):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        o = fn(); ttnn.synchronize_device(dev); dt = time.perf_counter() - t0
        if i >= WARM: ts.append(dt)
        if i < WARM + REPS - 1: ttnn.deallocate(o)
    return statistics.median(ts), o

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    dev = get_device(); L1 = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, ttnn.BufferType.L1)
    N, C = 512, 32
    t = torch.randn(1, N, N, C, dtype=torch.bfloat16)
    x = ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=L1)
    rows = []
    ms_a, o = med(lambda: ttnn.permute(x, (0, 3, 2, 1), memory_config=L1), dev)
    ref = ttnn.to_torch(o); ttnn.deallocate(o)
    rows.append(dict(arm="stock permute (0,3,2,1)", ms=1e3 * ms_a))

    def decomposed():
        m = rp.reblock_permute(x, L1)
        r = ttnn.transpose(m, -2, -1, memory_config=L1)
        ttnn.deallocate(m)
        return r
    ms_b, o = med(decomposed, dev)
    got = ttnn.to_torch(o); ttnn.deallocate(o)
    rows.append(dict(arm="kernel (0,3,1,2) + transpose(-2,-1)", ms=1e3 * ms_b,
                     speedup=ms_a / ms_b, bit_exact=bool(torch.equal(ref, got))))

    def decomposed_stock():
        m = ttnn.permute(x, (0, 3, 1, 2), memory_config=L1)
        r = ttnn.transpose(m, -2, -1, memory_config=L1)
        ttnn.deallocate(m)
        return r
    ms_c, o = med(decomposed_stock, dev)
    got = ttnn.to_torch(o); ttnn.deallocate(o)
    rows.append(dict(arm="stock (0,3,1,2) + transpose(-2,-1)", ms=1e3 * ms_c,
                     speedup=ms_a / ms_c, bit_exact=bool(torch.equal(ref, got))))

    y = ttnn.from_torch(torch.randn(1, C, N, N, dtype=torch.bfloat16), layout=ttnn.TILE_LAYOUT,
                        device=dev, dtype=ttnn.bfloat16, memory_config=L1)
    ms_d, o = med(lambda: ttnn.transpose(y, -2, -1, memory_config=L1), dev)
    ttnn.deallocate(o)
    rows.append(dict(arm="transpose(-2,-1) alone [1,32,512,512] L1", ms=1e3 * ms_d))
    for r in rows: print(json.dumps(r), flush=True)
    json.dump(dict(host="qb2", card=0, ttnn="0.68.0", uptime=up, n=N, c=C, warm=WARM, reps=REPS,
                   rows=rows), open(a.out, "w"), indent=1)
    print("wrote", a.out)
