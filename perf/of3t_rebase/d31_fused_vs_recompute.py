#!/usr/bin/env python3
"""D31: the tape's forward value and the function its backward differentiates are not the same
op. How far apart are they, and which of the two is right?

`taped_ttnn._v_sdpa` computes the forward with the production fused kernel,
`out_v = shipped(*ra, **rk)`, and hands it to `ag.triangle_attention(..., value=out_v)`. That
function's own docstring says the backward "reads q, k, v and bias and RECOMPUTES the scores; it
never reads the forward output". The recompute runs at `precise_config()`. So the value that
flows forward is the fused kernel's and the derivative that flows back is the precise
recompute's.

The existing record stops at "they differ by 3.3e-02, which is the fused softmax's own known
deficit". That is a gap, not a direction: it does not say which of the two is the deficient one,
and the answer decides whether D31 is a defect or a free improvement. If the recompute is the
accurate one, the tape differentiates a BETTER function than the forward computes, and the
gradient is not the thing to blame for a backward-specific error.

So this measures three things against a float64 CPU reference of the same composite:

    fused    vs float64   -- how wrong the value that flows forward is
    recompute vs float64  -- how wrong the function being differentiated is
    fused    vs recompute -- the gap the record already knows

CONVENTIONS, because getting them wrong would manufacture the finding. The fused kernel computes
`softmax((q k^T + mask) * scale) @ v`; `ag.triangle_attention` computes
`softmax(q k^T * scale + bias) @ v`. `_v_sdpa` reconciles them by scaling the bias on the tape
before the call, so the two composites are the same function and this replicates that exactly.
And amendment 10's warning is honoured: the object under the tape is the STOCK ttnn fused SDPA
that the verb wraps, not `triatt_sdpa`, which declines while `ops.taping()` is true.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--b", type=int, default=64, help="leading axis: S for triangle attention")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--bias-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path,
                    default=Path("perf/of3t_rebase/d31_fused_vs_recompute.json"))
    a = ap.parse_args()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device

    dev = get_device()
    B, H, N, D = a.b, a.heads, a.n, a.dim
    scale = D ** -0.5
    g = torch.Generator().manual_seed(a.seed)
    q_t = torch.randn(B, H, N, D, generator=g)
    k_t = torch.randn(B, H, N, D, generator=g)
    v_t = torch.randn(B, H, N, D, generator=g)
    b_t = torch.randn(1, H, N, N, generator=g) * a.bias_scale

    def to_dev(x, dt=ttnn.bfloat16):
        return ttnn.from_torch(x, dtype=dt, layout=ttnn.TILE_LAYOUT, device=dev)

    # float64 reference of the composite the fused kernel computes.
    ref = (torch.softmax((q_t.double() @ k_t.double().transpose(-1, -2) + b_t.double()) * scale,
                         dim=-1) @ v_t.double())

    qd, kd, vd, bd = (to_dev(x) for x in (q_t, k_t, v_t, b_t))

    # 1) the value that flows forward: the stock fused SDPA, called the way the verb calls it.
    fused = ttnn.transformer.scaled_dot_product_attention(qd, kd, vd, attn_mask=bd,
                                                          is_causal=False, scale=scale)
    fused_t = ttnn.to_torch(fused).double()

    # 2) the function the backward differentiates: the same node with value=None, so
    #    `ag.triangle_attention` computes its own forward with the precise recompute path.
    #    The bias is pre-scaled exactly as `_v_sdpa` does with `ag.scale`.
    bs = to_dev(b_t * scale)
    rec = ag.triangle_attention(ag.Tensor(qd), ag.Tensor(kd), ag.Tensor(vd), ag.Tensor(bs),
                                scale=scale, value=None)
    rec_t = ttnn.to_torch(rec.value if hasattr(rec, "value") else rec).double()

    def rel(x, y):
        return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))

    rep = {
        "instrument": "D31: the tape's forward value against the function its backward "
                      "differentiates, both against a float64 reference",
        "shape": {"B": B, "heads": H, "n": N, "dim": D, "scale": scale,
                  "bias_scale": a.bias_scale},
        "dtype_device": "bfloat16 in, fused kernel default out",
        "object_under_test": "ttnn.transformer.scaled_dot_product_attention, the stock fused "
                             "SDPA taped_ttnn._v_sdpa wraps -- NOT triatt_sdpa, which declines "
                             "while ops.taping() is true",
        "fused_vs_float64": rel(fused_t, ref),
        "recompute_vs_float64": rel(rec_t, ref),
        "fused_vs_recompute": rel(fused_t, rec_t),
    }
    rep["verdict"] = (
        "the tape differentiates the MORE accurate of the two"
        if rep["recompute_vs_float64"] < rep["fused_vs_float64"] else
        "the tape differentiates the LESS accurate of the two")
    rep["ratio_fused_over_recompute"] = (rep["fused_vs_float64"]
                                         / (rep["recompute_vs_float64"] or 1e-300))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1) + "\n")
    print(f"fused      vs float64 : {rep['fused_vs_float64']:.4e}")
    print(f"recompute  vs float64 : {rep['recompute_vs_float64']:.4e}")
    print(f"fused      vs recompute: {rep['fused_vs_recompute']:.4e}")
    print(f"-> {rep['verdict']} ({rep['ratio_fused_over_recompute']:.2f}x)")
    print("->", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
