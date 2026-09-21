#!/usr/bin/env python3
"""Why the renormalisation is a repair on the device and a no-op on the host path.

`d_logits = y (g - sum_j g_j y_j)` has exactly vanishing row sums when `sum_j y_j = 1`, and the
attention backward downstream depends on that. of3t-apbgrad found `ttnn.softmax` does not give
rows that sum to one and repaired it with `inner = sum(g y) / sum(y)`.

This probe measures both halves of that sentence on the OF3 diffusion transformer's own softmax
shape: what each softmax's row sums actually are, and what the resulting `d_logits` row sums are
with the repair off and on. The host float64 arm is the control -- if its rows already sum to one
then the repair cannot change its gradient, and AMENDMENT 1's Arm B is a no-op by arithmetic
rather than by luck.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import ttnn


def stats(t):
    t = t.to(torch.float64)
    return {"max_abs_dev_from_one": float((t - 1.0).abs().max()),
            "mean": float(t.mean()), "min": float(t.min()), "max": float(t.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1x16x384x384")
    ap.add_argument("--spread", type=float, default=8.0)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    shape = tuple(int(v) for v in a.shape.split("x"))

    from tt_bio.main import ensure_p300_mesh_descriptor
    ensure_p300_mesh_descriptor()
    from tt_bio.autograd import host_f64_softmax_values
    from tt_bio.tenstorrent import get_device
    dev = get_device()

    torch.manual_seed(0)
    host = torch.randn(*shape, dtype=torch.float32) * a.spread
    g = torch.randn(*shape, dtype=torch.float32)
    x = ttnn.from_torch(host, dtype=ttnn.float32, layout=ttnn.TILE_LAYOUT, device=dev)

    y_dev = ttnn.to_torch(ttnn.softmax(x, dim=-1)).double()
    y_hst = ttnn.to_torch(host_f64_softmax_values(x, -1)[1]).double()
    y_f64 = torch.softmax(host.double(), dim=-1)
    gd = g.double()

    rep = {"shape": list(shape), "spread": a.spread,
           "row_sums": {"ttnn.softmax": stats(y_dev.sum(-1)),
                        "host_f64_softmax, as returned to the card (fp32)":
                            stats(y_hst.sum(-1)),
                        "host_f64_softmax, the float64 value the backward reads":
                            stats(y_f64.sum(-1))}}

    # d_logits row sums, the quantity the repair exists to make vanish.
    out = {}
    for name, y in (("ttnn.softmax", y_dev), ("host float64", y_f64)):
        inner = (gd * y).sum(-1, keepdim=True)
        plain = (y * (gd - inner)).sum(-1)
        renorm = (y * (gd - inner / y.sum(-1, keepdim=True))).sum(-1)
        out[name] = {"d_logits_row_sum_max_abs_plain": float(plain.abs().max()),
                     "d_logits_row_sum_max_abs_renormalised": float(renorm.abs().max())}
    rep["d_logits"] = out
    print(json.dumps(rep, indent=1), flush=True)
    if a.out:
        json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True)
        print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
