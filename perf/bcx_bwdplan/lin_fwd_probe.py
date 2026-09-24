#!/usr/bin/env python3
"""Forward `ttnn.linear` on a one-row-per-batch operand [256,1,K]: rank-3 as issued vs folded to 2-D."""
import json, statistics, sys, time, pathlib
import torch
ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from perf.bcx_stack.stack import Clock  # noqa: E402

import ttnn
from tt_bio import tenstorrent as tt, af2
dev = tt.get_device()
clock = Clock()
cfg = af2.compute_kernel_config()
g = torch.Generator().manual_seed(0)
up = lambda x: ttnn.from_torch(x.to(torch.bfloat16), dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
res = {}
for K, N in ((256, 768), (256, 256), (256, 1024), (1024, 256)):
    x = up(torch.randn(256, 1, K, generator=g))
    w = up(torch.randn(K, N, generator=g) / 16)
    b = up(torch.randn(1, N, generator=g))
    ref = (ttnn.to_torch(x).double().reshape(256, K) @ ttnn.to_torch(w).double()
           + ttnn.to_torch(b).double().reshape(1, N))
    arms = {"rank3": lambda: ttnn.linear(x, w, bias=b, compute_kernel_config=cfg,
                                         core_grid=tt.CORE_GRID_MAIN),
            "flat": lambda: ttnn.reshape(ttnn.linear(ttnn.reshape(x, [256, K]), w, bias=b,
                                                     compute_kernel_config=cfg,
                                                     core_grid=tt.CORE_GRID_MAIN), [256, 1, N])}
    out, base = {}, None
    for name, fn in arms.items():
        y = ttnn.to_torch(fn()).double().reshape(256, N)
        base = y if base is None else base
        for _ in range(3):
            fn()
        ttnn.synchronize_device(dev)
        ts, t0w = [], time.time()
        while time.time() - t0w < 1.0:
            t0 = time.perf_counter()
            for _ in range(20):
                fn()
            ttnn.synchronize_device(dev)
            ts.append((time.perf_counter() - t0) / 20)
        out[name] = {"us": statistics.median(ts) * 1e6,
                     "rel_l2_vs_f64": float((y - ref).norm() / ref.norm()),
                     "bitexact_vs_rank3": bool(torch.equal(y, base)),
                     "aiclk": clock.window([(t0w, time.time())])}
        print(K, N, name, out[name], flush=True)
    res[f"{K}x{N}"] = out
clock.stop()
out = pathlib.Path(sys.argv[1])
out.write_text(json.dumps(res, indent=1))
print("wrote", out)
