"""Time the structure-module seam's pair copy: [224, 224, 128] fp32 up (from_torch) and down (to_torch)."""
import json, sys, time
import numpy as np, torch
from tt_bio import tenstorrent as T
import ttnn
dev = T.get_device()
x = torch.randn(224, 224, 128)
res = {}
for nm, dt in (("fp32", ttnn.float32), ("bf16", ttnn.bfloat16)):
    up, dn = [], []
    for _ in range(6):
        t0 = time.perf_counter(); y = ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev); ttnn.synchronize_device(dev); up.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); z = ttnn.to_torch(y); dn.append(time.perf_counter() - t0)
        ttnn.deallocate(y)
    res[nm] = {"up_ms": sorted(up)[1:-1], "down_ms": sorted(dn)[1:-1]}
    res[nm] = {k: [round(v * 1e3, 2) for v in vs] for k, vs in res[nm].items()}
print(json.dumps(res)); json.dump(res, open(sys.argv[1], "w"))
