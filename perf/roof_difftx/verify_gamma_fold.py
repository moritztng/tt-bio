#!/usr/bin/env python3
"""Does folding the AdaLN s_norm gain into the projection weight cost accuracy?

The conditioning concatenation needs every layer's four AdaLN projections to read ONE shared
input, and they do not: each reads its own `LayerNorm(s, weight=gamma_i)`. Since
`nn.LayerNorm(dim, bias=False)` is `gamma_i * s_hat` with `s_hat` the parameter-free
normalisation, `LN_i(s) @ W_i == s_hat @ (gamma_i[:, None] * W_i)` in exact arithmetic. In bf16
it is a different rounding: the shipped path rounds `gamma_i * s_hat` per element of the
activation, the folded path rounds `gamma_i * W_i` per element of the weight, once, offline.

Reference is float64 throughout and never another approximation. Host model of the arithmetic,
not of ttnn's layer_norm kernel: the question is the ORDER of the two roundings, which is the
part that differs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
DIM = 768
S = 512


def rel_rms(x, ref):
    return float(torch.sqrt(torch.mean((x - ref) ** 2) / torch.mean(ref ** 2)))


def pcc(x, ref):
    a = (x - x.mean()).flatten()
    b = (ref - ref.mean()).flatten()
    return float((a @ b) / (a.norm() * b.norm()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "verify_gamma_fold.json")
    ap.add_argument("--trials", type=int, default=8)
    a = ap.parse_args()
    torch.manual_seed(0)

    rows = []
    for k in range(a.trials):
        s = torch.randn(S, DIM, dtype=torch.float64)
        gamma = 1.0 + 0.1 * torch.randn(DIM, dtype=torch.float64)
        w = (0.02 * torch.randn(DIM, DIM, dtype=torch.float64))

        mu = s.mean(-1, keepdim=True)
        var = s.var(-1, unbiased=False, keepdim=True)
        s_hat = (s - mu) / torch.sqrt(var + 1e-5)
        ref = (gamma * s_hat) @ w                                  # float64 truth

        b = torch.bfloat16
        ship = ((gamma * s_hat).to(b).double() @ w.to(b).double())  # gamma on the activation
        fold = (s_hat.to(b).double() @ (gamma[:, None] * w).to(b).double())  # gamma on the weight

        rows.append({"trial": k,
                     "shipped_rel_rms": rel_rms(ship, ref), "shipped_pcc": pcc(ship, ref),
                     "folded_rel_rms": rel_rms(fold, ref), "folded_pcc": pcc(fold, ref),
                     "folded_vs_shipped_rel_rms": rel_rms(fold, ship)})

    agg = {k: sum(r[k] for r in rows) / len(rows)
           for k in ("shipped_rel_rms", "folded_rel_rms", "folded_vs_shipped_rel_rms",
                     "shipped_pcc", "folded_pcc")}
    out = {"dim": DIM, "tokens": S, "trials": a.trials, "reference": "float64",
           "mean": agg, "rows": rows}
    a.out.write_text(json.dumps(out, indent=1))
    print("against a float64 reference, mean over %d trials:" % a.trials)
    print("  shipped  gamma on the activation   rel RMS %.6e   PCC %.9f"
          % (agg["shipped_rel_rms"], agg["shipped_pcc"]))
    print("  folded   gamma on the weight       rel RMS %.6e   PCC %.9f"
          % (agg["folded_rel_rms"], agg["folded_pcc"]))
    print("  folded vs shipped                  rel RMS %.6e" % agg["folded_vs_shipped_rel_rms"])
    print("  the fold costs %.4fx the shipped path's own distance from the truth"
          % (agg["folded_rel_rms"] / agg["shipped_rel_rms"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
