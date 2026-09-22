#!/usr/bin/env python3
"""A FULLY MASKED row is where the 5-op chain overflows, and why.

`ttnn.max` is not exact: on [1,14,4,32,128] it reads 3.4e-03 relative against the true row max.
That is harmless when the row has a real entry, because the max is then O(1) and the error is
O(1e-3). On a row where every key is masked the row max is the mask value itself, so a 0.3 %
relative error on -1e9 is an absolute error of millions -- and `d = x - m` comes out POSITIVE
by millions, which `exp` turns into inf. The fused kernel does not subtract a max it computed
this way, so it survives the same row.

The repair is one op and it is exact for a true max: clamp `d` at 0. A softmax is invariant to
the row shift, so this changes nothing where the max is right and removes the overflow where
the reduction is short.
"""
import json
import sys

import torch
import ttnn

sys.path.insert(0, "/home/ttuser/.coworker/wt/of3t-softgrad")
from tt_bio.tenstorrent import get_device, _accurate_softmax, _SOFTMAX_PRECISE_CKC

dev = get_device()


def chain(xt, clamp):
    m = ttnn.max(xt, dim=-1, keepdim=True)
    d = ttnn.subtract(xt, m)
    if clamp:
        d = ttnn.maximum(ttnn.minimum(d, 0.0), float(clamp))
    ttnn.exp(d, output_tensor=d)
    s = ttnn.sum(d, dim=-1, keepdim=True, compute_kernel_config=_SOFTMAX_PRECISE_CKC)
    return ttnn.divide(d, s)


torch.manual_seed(0)
shape = [1, 14, 4, 32, 128]
x = torch.randn(*shape) * 3.0
x[:, 7:, :, :, :] -= 1e9          # blocks 7..13: every key masked, the real case
xt = ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)

mx = ttnn.to_torch(ttnn.max(xt, dim=-1, keepdim=True)).double().reshape(x.shape[:-1] + (1,))
want = x.double().amax(-1, keepdim=True)
gap = (mx - want)
rep = {
    "shape": shape,
    "fully_masked_blocks": 7,
    "ttnn_max_vs_true_max": {
        "rel_l2": float((mx - want).norm() / want.norm()),
        "worst_absolute_shortfall": float((want - mx).max()),
        "worst_absolute_overshoot": float(gap.max()),
        "on_masked_rows_worst_shortfall": float((want - mx)[:, 7:].max()),
        "on_real_rows_worst_shortfall": float((want - mx)[:, :7].max()),
    },
}
ref = torch.softmax(x.double(), dim=-1)
for name, y in (("fused", ttnn.softmax(xt, dim=-1, compute_kernel_config=_SOFTMAX_PRECISE_CKC)),
                ("accurate_shipped", _accurate_softmax(xt, compute_kernel_config=_SOFTMAX_PRECISE_CKC)),
                ("accurate_clamp88", chain(xt, clamp=-88.0)),
                ("accurate_clamp60", chain(xt, clamp=-60.0)),
                ("accurate_clamp30", chain(xt, clamp=-30.0))):
    t = ttnn.to_torch(y).double()
    fin = bool(torch.isfinite(t).all())
    rep[name] = {"finite": fin, "n_nonfinite": int((~torch.isfinite(t)).sum()),
                 "rel_l2": (float((t - ref).norm() / ref.norm()) if fin else None)}
    print(name, json.dumps(rep[name]), flush=True)
print("max gap", json.dumps(rep["ttnn_max_vs_true_max"]), flush=True)
json.dump(rep, open("/home/ttuser/.coworker/wt/of3t-softgrad/perf/of3t_softgrad/"
                    "FULLY_MASKED_ROW_OVERFLOW.json", "w"), indent=1, sort_keys=True)
print("wrote FULLY_MASKED_ROW_OVERFLOW.json")
