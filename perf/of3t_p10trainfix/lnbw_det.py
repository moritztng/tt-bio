#!/usr/bin/env python3
"""Is `_layer_norm_bw`'s dx a function of its inputs? Repeat it on fixed inputs, DRAM dirtied between.

`lnhash.py` found two calls in one training step, diffusion conditioning's transition layer
norms at [1,384,384,128] and [1,384,384] bf16, where two runs handed the backward bit-identical
g, x and gamma, got bit-identical dgamma and dbeta back, and a dx 7-40 % apart in norm. This
isolates the closure: the same inputs, K repeats, garbage written to freshly freed DRAM between
them, and every intermediate of the dx half hashed so a divergence names its op.

    TT_VISIBLE_DEVICES=<card> lnbw_det.py --shape 1,384,384,128 --reps 4
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tt_bio.main import ensure_p300_mesh_descriptor                  # noqa: E402
ensure_p300_mesh_descriptor()

import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402
import ttnn                                                           # noqa: E402
from tt_bio import autograd as ag                                     # noqa: E402
from tt_bio.tenstorrent import get_device                             # noqa: E402


def h(v):
    a = ttnn.to_torch(v).float().numpy()
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12], float(np.linalg.norm(a))


def dirty(dev, gb):
    """Fill `gb` of DRAM with noise and free it, so a read of unwritten memory reads noise."""
    ts = []
    for _ in range(int(gb * 4)):
        t = torch.randn(1, 1024, 1024, 64) * 1e3                       # 256 MB in bf16... ~128 MB
        ts.append(ttnn.from_torch(t, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev))
    for t in ts:
        ttnn.deallocate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="1,384,384,128")
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--dirty-gb", type=float, default=4)
    ap.add_argument("--g-dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--x-dtype", choices=("bf16", "fp32"), default="bf16")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    shape = [int(s) for s in a.shape.split(",")]
    K = shape[-1]
    dev = get_device()
    torch.manual_seed(0)
    bf = dict(dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=dev)
    DT = {"bf16": ttnn.bfloat16, "fp32": ttnn.float32}
    xv = ttnn.from_torch(torch.randn(shape) * 3 + 1, **{**bf, "dtype": DT[a.x_dtype]})
    gv = ttnn.from_torch(torch.randn(shape) * 0.01, **{**bf, "dtype": DT[a.g_dtype]})
    gam = ttnn.from_torch(torch.randn(K) * 0.5 + 1, **bf)
    bet = ttnn.from_torch(torch.randn(K) * 0.1, **bf)
    cfg = ag.precise_config()
    rows = []
    for r in range(a.reps):
        if r:
            dirty(dev, a.dirty_gb)
        # the closure, op by op, as `_layer_norm_bw` writes it
        mean = ag._row_mean(xv, cfg, K)
        centered = ttnn.subtract(xv, mean)
        var = ag._row_mean(ttnn.multiply(centered, centered), cfg, K)
        rstd = ttnn.rsqrt(ttnn.add(var, 1e-5))
        norm = ttnn.multiply(centered, rstd)
        dnorm = ttnn.multiply(gv, gam)
        dn_mean = ag._row_mean(dnorm, cfg, K)
        dnn = ttnn.multiply(dnorm, norm)
        dn_norm_mean = ag._row_mean(dnn, cfg, K)
        t1 = ttnn.subtract(dnorm, dn_mean)
        t2 = ttnn.multiply(norm, dn_norm_mean)
        dx0 = ttnn.subtract(t1, t2)
        dx = ttnn.multiply(dx0, rstd)
        # and through the real closure
        X = ag.Tensor(xv, requires_grad=True)
        G = ag.Tensor(gam, requires_grad=True)
        B = ag.Tensor(bet, requires_grad=True)
        ag._layer_norm_bw(X, G, B, 1e-5, cfg)(gv)
        row = {k: h(v) for k, v in (("mean", mean), ("centered", centered), ("var", var),
                                     ("rstd", rstd), ("norm", norm), ("dnorm", dnorm),
                                     ("dn_mean", dn_mean), ("dnn", dnn),
                                     ("dn_norm_mean", dn_norm_mean), ("t1", t1), ("t2", t2),
                                     ("dx0", dx0), ("dx", dx), ("closure_dx", X.grad),
                                     ("closure_dgamma", G.grad), ("closure_dbeta", B.grad))}
        rows.append(row)
        print(r, {k: v[0] for k, v in row.items()}, flush=True)
    diff = [k for k in rows[0] if len({r[k][0] for r in rows}) > 1]
    print("DIFFER:", diff)
    for k in diff:
        print("  ", k, [r[k][1] for r in rows])
    if a.out:
        a.out.write_text(json.dumps({"shape": shape, "reps": rows, "differ": diff}, indent=1))


if __name__ == "__main__":
    main()
