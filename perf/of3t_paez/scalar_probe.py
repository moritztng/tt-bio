#!/usr/bin/env python3
"""of3t-paez: does ttnn.multiply(fp32 tensor, python float) round the scalar to bf16?

The device denoise arm's EDM output is `xl_noisy * c_skip + rl_update * c_out` with host-float
scalars (openfold3_diffusion_module.py). At this step's sigma, bf16 rounds c_skip 0.99963923 to
1.0 and c_out 0.30390167 to 0.3046875. Read back x * c for x = 1 and x = random, fp32 tile.
"""
import json
import sys

import torch
import ttnn

C = {"c_skip": 0.9996392335070935, "c_out": 0.30390166531968693}
from tt_bio.tenstorrent import get_device
dev = get_device()
rec = {}
try:
    x = torch.randn(1, 384, 32, dtype=torch.float32) * 10
    for name, c in C.items():
        t = ttnn.from_torch(x, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)
        y = ttnn.to_torch(ttnn.multiply(t, c)).double()
        eff = float((y * x.double()).sum() / (x.double() ** 2).sum())
        rec[name] = {"host": c, "effective": eff, "bf16": float(torch.tensor(c).bfloat16().double()),
                     "rel_err_vs_host": (eff - c) / c,
                     "max_abs_vs_fp32_product": float((y - (x * torch.tensor(c, dtype=torch.float32)).double()).abs().max())}
finally:
    pass
print(json.dumps(rec, indent=1))
json.dump(rec, open(sys.argv[1], "w"), indent=1)
