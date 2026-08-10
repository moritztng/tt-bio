#!/usr/bin/env python3
"""X3 deliverable 2: the single-core untilize fallback fires on shapes a real fold reaches
(256-token target: z is 256x256 tiles). Can row-blocking walk the untilize out of the fallback
window, and is it bit-exact?"""
from __future__ import annotations
import json, statistics as st, sys, time
import torch, ttnn
from tt_bio.tenstorrent import get_device

DRAM = ttnn.DRAM_MEMORY_CONFIG
dev = get_device()
log = lambda *a: print(*a, file=sys.stderr, flush=True)


def timed(fn, reps=3):
    fn()
    ttnn.synchronize_device(dev)
    o = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        fn()
        ttnn.synchronize_device(dev)
        o.append(time.perf_counter() - t0)
    return st.median(o) * 1e6


def blocked(t, blk_tiles):
    """Untilize in row blocks of `blk_tiles` tile-rows, then concat in row-major."""
    parts = []
    rows = t.shape[0]
    step = blk_tiles * 32
    for s in range(0, rows, step):
        parts.append(ttnn.to_layout(t[s:min(s + step, rows), :], ttnn.ROW_MAJOR_LAYOUT))
    out = parts[0] if len(parts) == 1 else ttnn.concat(parts, dim=0)
    if len(parts) > 1:
        for p in parts:
            ttnn.deallocate(p)
    return out


res = {}
for rt, ct, tag in ((256, 256, "T=256 tokens (256x256 tiles, the fallback fires)"),
                    (298, 298, "T=298 tokens (298x298 tiles, control, already fast)")):
    x = torch.randn(rt * 32, ct * 32, dtype=torch.bfloat16)
    t = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16,
                        memory_config=DRAM)
    nb = rt * ct * 2048
    row = {"MB": round(nb / 2**20, 1)}
    base_us = timed(lambda: ttnn.deallocate(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT)))
    row["whole"] = {"us": round(base_us, 1), "GB/s": round(2 * nb / (base_us * 1e-6) / 1e9, 1)}
    log(f"{tag}")
    log(f"  whole                 {base_us:10.1f} us  {2*nb/(base_us*1e-6)/1e9:7.1f} GB/s")
    ref = ttnn.to_torch(ttnn.to_layout(t, ttnn.ROW_MAJOR_LAYOUT))
    for blk in (32, 64, 128):
        us = timed(lambda b=blk: ttnn.deallocate(blocked(t, b)))
        got = ttnn.to_torch(blocked(t, blk))
        eq = bool(torch.equal(ref, got))
        row[f"blocked_{blk}"] = {"us": round(us, 1), "GB/s": round(2 * nb / (us * 1e-6) / 1e9, 1),
                                 "speedup": round(base_us / us, 2), "torch_equal": eq}
        log(f"  row blocks of {blk:3d} tiles {us:10.1f} us  "
            f"{2*nb/(us*1e-6)/1e9:7.1f} GB/s  {base_us/us:6.2f}x  torch.equal={eq}")
    res[tag] = row
    ttnn.deallocate(t)

print(json.dumps(res, indent=1))
