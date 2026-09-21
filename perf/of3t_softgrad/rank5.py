#!/usr/bin/env python3
"""`_accurate_softmax` on the atom encoder's rank-5 block-sparse attention shape.

The scope arm's first non-finite forward activation is a softmax output of shape
[1, 14, 4, 32, 128] -- the atom encoder's (blocks, heads, queries, keys) layout. The fused
`ttnn.softmax` is finite there and the 5-op chain is not, so the question is whether the chain
survives a rank-5 last-axis reduction at all, independently of the model.
"""
import json
import sys

import torch
import ttnn

sys.path.insert(0, "/home/ttuser/.coworker/wt/of3t-softgrad")
from tt_bio.tenstorrent import get_device, _accurate_softmax, _SOFTMAX_PRECISE_CKC

dev = get_device()
torch.manual_seed(0)
out = {}
for shape in ([1, 14, 4, 32, 128], [1, 4, 32, 128], [14 * 4, 32, 128], [1, 16, 384, 384]):
    x = torch.randn(*shape) * 3.0
    if len(shape) == 5:                      # the real thing carries a key mask
        x[..., 64:] -= 1e9
    xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
    ref = torch.softmax(x.double(), dim=-1)
    row = {"shape": shape}
    for name, fn in (("fused", lambda t: ttnn.softmax(t, dim=-1,
                                                      compute_kernel_config=_SOFTMAX_PRECISE_CKC)),
                     ("accurate", lambda t: _accurate_softmax(
                         t, compute_kernel_config=_SOFTMAX_PRECISE_CKC))):
        try:
            y = ttnn.to_torch(fn(xt)).double()
            fin = bool(torch.isfinite(y).all())
            row[name] = {
                "finite": fin,
                "n_nonfinite": int((~torch.isfinite(y)).sum()),
                "rel_l2": (float((y - ref).norm() / ref.norm()) if fin else None),
                "row_sum_min": (float(y.sum(-1).min()) if fin else None),
                "row_sum_max": (float(y.sum(-1).max()) if fin else None),
            }
        except Exception as e:
            row[name] = {"raised": f"{type(e).__name__}: {e}"}
    print(shape, json.dumps(row["fused"]), json.dumps(row["accurate"]), flush=True)
    out[str(shape)] = row

# and the reduction the chain is built from, on the same rank-5 shape
x = torch.randn(1, 14, 4, 32, 128) * 3.0
xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
prim = {}
for nm, fn in (("max", lambda t: ttnn.max(t, dim=-1, keepdim=True)),
               ("sum", lambda t: ttnn.sum(t, dim=-1, keepdim=True,
                                          compute_kernel_config=_SOFTMAX_PRECISE_CKC))):
    try:
        y = ttnn.to_torch(fn(xt)).double()
        want = (x.double().amax(-1, keepdim=True) if nm == "max"
                else x.double().sum(-1, keepdim=True))
        prim[nm] = {"shape_out": list(y.shape), "finite": bool(torch.isfinite(y).all()),
                    "rel_l2": float((y.reshape(want.shape) - want).norm() / want.norm())}
    except Exception as e:
        prim[nm] = {"raised": f"{type(e).__name__}: {e}"}
    print("primitive", nm, json.dumps(prim[nm]), flush=True)
out["rank5_primitives"] = prim
json.dump(out, open("/home/ttuser/.coworker/wt/of3t-softgrad/perf/of3t_softgrad/"
                    "RANK5_ACCURATE_SOFTMAX.json", "w"), indent=1, sort_keys=True)
print("wrote RANK5_ACCURATE_SOFTMAX.json")
