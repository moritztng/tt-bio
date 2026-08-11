#!/usr/bin/env python3
"""Settle the 4.8x disagreement on tenstorrent.py:1709.

The 512 aa census prices the (0,3,2,1) chunk move at 0.200 ms; an isolated probe measured 0.955 ms
at the same shape, dtype and buffers. 16 calls per block, so the two readings differ by 5.8 s/fold
and they decide whether `_transform_chunk`'s `decompose` guard is the largest bit-exact lever in
the program or nothing at all. Same process, minimal residency, every rep printed.
"""
import argparse, json, statistics, subprocess, sys, time
from pathlib import Path
import torch, ttnn
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from tt_bio.tenstorrent import get_device            # noqa: E402
WARM, REPS = 3, 7

def arm(dev, x, dims, mc):
    ts, o = [], None
    for i in range(WARM + REPS):
        ttnn.synchronize_device(dev); t0 = time.perf_counter()
        o = ttnn.permute(x, dims, memory_config=mc); ttnn.synchronize_device(dev)
        dt = time.perf_counter() - t0
        if i >= WARM: ts.append(1e3 * dt)
        if i < WARM + REPS - 1: ttnn.deallocate(o)
    ttnn.deallocate(o)
    return dict(dims=list(dims), buf=mc.buffer_type.name, ms_med=statistics.median(ts),
                ms_min=min(ts), ms_max=max(ts), reps=[round(t, 4) for t in ts])

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args()
    up = subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()
    dev = get_device()
    rows = []
    for buf in (ttnn.BufferType.L1, ttnn.BufferType.DRAM):
        mc = ttnn.MemoryConfig(ttnn.TensorMemoryLayout.INTERLEAVED, buf)
        x = ttnn.from_torch(torch.randn(1, 512, 512, 32, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16, memory_config=mc)
        for dims in ((0, 3, 1, 2), (0, 3, 2, 1)):
            r = arm(dev, x, dims, mc); rows.append(r); print(json.dumps(r), flush=True)
        ttnn.deallocate(x)
    json.dump(dict(host="qb2", card=0, ttnn="0.68.0", uptime=up, shape=[1, 512, 512, 32],
                   dtype="bfloat16", warm=WARM, reps=REPS, rows=rows), open(a.out, "w"), indent=1)
    print("wrote", a.out)
