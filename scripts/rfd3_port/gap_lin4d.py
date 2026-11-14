"""RFD3's linears at their census shapes: ttnn's default plan (what the model calls) against an explicit core grid
and against the same call on the input flattened to [1, 1, M, K], each against float64 and against the default.

    python scripts/rfd3_port/gap_lin4d.py OUT.json [reps]

cen3 (WH fast, @1000 MHz) has the 4-D ones at 3-5x their DRAM roof (1x685x32x768 @ 768x128: 0.82 ms vs 0.16), and
a plan-less batched matmul was worth 3-14x on BindCraft 2. Per shape and arm: ms/call (median of reps, each a burst
of calls between two syncs so dispatch is amortised), rel_rms and max_abs to float64, max_abs to the default output.
"""
import json
import os
import statistics
import sys
import time

import torch

from tt_bio.main import ensure_p300_mesh_descriptor
ensure_p300_mesh_descriptor()
import ttnn  # noqa: E402
from tt_bio import tenstorrent as T  # noqa: E402
from tt_bio.rfd3 import model as M  # noqa: E402

out_path, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 7
dev = T.get_device()
ckc = M._default_compute_kernel_config()
g = dev.compute_with_storage_grid_size()
grid = ttnn.CoreGrid(y=g.y, x=g.x)
SHAPES = [  # (x shape, K -> N, bias), calls per step in cen3 WH fast
    ((1, 685, 704, 128), 32, False),     # token pair bias, 4
    ((1, 685, 32, 768), 128, False),     # decoder / downcast_q, 6
    ((1, 685, 32, 256), 128, False),     # decoder, 12
    ((1, 685, 32, 128), 128, False),     # decoder, 16
    ((1, 685, 32, 128), 768, True),      # decoder / downcast_q, 3
    ((1, 6051, 128, 32), 32, False),     # atom pair bias, 2
    ((1, 6080, 128), 128, False),        # atom, 63
    ((1, 6080, 128), 256, False),        # atom, 18
    ((1, 704, 768), 1024, False),        # DiT, 144
    ((1, 704, 768), 1536, False),        # DiT, 72
    ((1, 704, 384), 768, True),          # DiT, 144
]
rec = {"reps": reps, "card": os.environ.get("TT_VISIBLE_DEVICES"), "arch": str(dev.arch()),
       "grid": [g.y, g.x], "cases": []}


def burst(fn, n):
    ts = []
    for i in range(reps + 1):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = [fn() for _ in range(n)]
        ttnn.synchronize_device(dev)
        if i:
            ts.append((time.perf_counter() - t0) / n)
        for o in outs:
            ttnn.deallocate(o)
    return round(1e3 * statistics.median(ts), 4)


gen = torch.Generator().manual_seed(0)
for xs, n_out, has_b in SHAPES:
    k = xs[-1]
    x = torch.randn(*xs, generator=gen).to(torch.bfloat16)
    w = (torch.randn(k, n_out, generator=gen) / k ** 0.5).to(torch.bfloat16)
    b = (0.1 * torch.randn(n_out, generator=gen)).to(torch.bfloat16) if has_b else None
    ref = x.double() @ w.double() + (b.double() if has_b else 0)
    xt = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    wt = ttnn.from_torch(w, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    bt = ttnn.from_torch(b.reshape(1, -1), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev) if has_b else None
    m = 1
    for d in xs[:-1]:
        m *= d
    flat = (1, 1, m, k)
    kw = dict(compute_kernel_config=ckc, dtype=ttnn.bfloat16)
    if has_b:
        kw["bias"] = bt
    arms = {
        "default": lambda: ttnn.linear(xt, wt, **kw),
        "grid": lambda: ttnn.linear(xt, wt, core_grid=grid, **kw),
        "flat": lambda: ttnn.reshape(ttnn.linear(ttnn.reshape(xt, flat), wt, **kw), xs[:-1] + (n_out,)),
        "flat_grid": lambda: ttnn.reshape(ttnn.linear(ttnn.reshape(xt, flat), wt, core_grid=grid, **kw),
                                          xs[:-1] + (n_out,)),
    }
    n_burst = max(1, min(50, int(2e8 // (x.numel() * 2 + m * n_out * 2))))
    case = {"x": list(xs), "w": [k, n_out], "bias": has_b, "burst": n_burst,
            "roof_ms": round(1e3 * (x.numel() * 2 + m * n_out * 2) / 248e9, 4), "arms": {}}
    base = None
    for name, fn in arms.items():
        try:
            o = fn()
            ot = ttnn.to_torch(o).reshape(ref.shape)
            ttnn.deallocate(o)
            d = ot.double() - ref
            a = {"ms": burst(fn, n_burst), "rel_rms": float(d.pow(2).mean().sqrt() / ref.pow(2).mean().sqrt()),
                 "max_abs": float(d.abs().max())}
            if base is None:
                base = ot
            a["vs_default_max_abs"] = float((ot.float() - base.float()).abs().max())
        except Exception as e:
            a = {"error": str(e)[:300]}
        case["arms"][name] = a
        print(xs, n_out, name, a, flush=True)
    rec["cases"].append(case)
    for t in (xt, wt, bt):
        if t is not None:
            ttnn.deallocate(t)
    with open(out_path, "w") as f:
        json.dump(rec, f, indent=1)
print("wrote", out_path)
