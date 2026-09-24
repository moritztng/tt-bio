"""torch.equal check of tt_bio.page_copy against the same slice assignment in torch.

    TT_VISIBLE_DEVICES=<c> ... python perf/mgx_wide_seq/page_copy_check.py
"""
import json
import time

import torch

import ttnn

from tt_bio import page_copy as PC
from tt_bio.tenstorrent import get_device

dev = get_device()
up = lambda t: ttnn.from_torch(t, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
rows = []
g = torch.Generator().manual_seed(0)
for S, C, kind, s, R in [(64, 32, "rows", 0, 64), (96, 64, "rows", 32, 17), (320, 384, "rows", 100, 3),
                         (320, 384, "cols", 64, 96), (320, 384, "cols", 288, 32),
                         (300, 64, "rows", 299, 1), (300, 64, "cols", 288, 12),
                         (1504, 384, "rows", 704, 64), (1504, 384, "cols", 1024, 224)]:
    z = torch.randn(1, S, S, C, generator=g).bfloat16()
    b = torch.randn(*((1, R, S, C) if kind == "rows" else (1, S, R, C)), generator=g).bfloat16()
    zt, bt = up(z), up(b)
    ttnn.synchronize_device(dev)
    t0 = time.perf_counter()
    (PC.write_rows if kind == "rows" else PC.write_cols)(zt, bt, s)
    ttnn.synchronize_device(dev)
    ms = (time.perf_counter() - t0) * 1e3
    ref = z.clone()
    if kind == "rows":
        ref[:, s:s + R] = b
    else:
        ref[:, :, s:s + R] = b
    got = ttnn.to_torch(zt)
    r = {"S": S, "C": C, "kind": kind, "start": s, "R": R, "equal": bool(torch.equal(got, ref)),
         "ms": round(ms, 3), "GB_per_s": round(b.numel() * 2 / ms / 1e6, 1)}
    rows.append(r)
    print(json.dumps(r), flush=True)
    ttnn.deallocate(zt)
    ttnn.deallocate(bt)
print("ALL_EQUAL" if all(r["equal"] for r in rows) else "MISMATCH")
