#!/usr/bin/env python3
"""Can the forward-model check tell a wrong function from bf16 rounding?

`census.py`'s reading -- `refvjp`'s modelled forward against the device's own `out_v`, worst
3.4986e-03 over 88 live firings -- is only evidence if a WRONG model would have read larger.
0.0035 is 2^-8.2 and bf16's unit roundoff is 2^-8 = 3.9e-03, so the number on its own is equally
consistent with "the right function at bf16" and with "some other function that happens to be
close". This separates the two by scoring plausible wrong models against the correct one on the
same inputs, in float64, at the shapes the census actually fires.

A candidate that scores BELOW the device's own bf16 error is one this check cannot exclude, and
it is reported as such rather than left out.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import refvjp as R                                                     # noqa: E402

torch.manual_seed(20260922)
EPS = 1e-5
DEV_BF16 = 3.4985898389263037e-03          # census.py's worst, SIDE_C64.json


def rel(a, b):
    return float(torch.linalg.vector_norm((a - b).reshape(-1))
                 / torch.linalg.vector_norm(b.reshape(-1)))


def ln_variants(x, gamma, beta):
    ref = R.layer_norm_forward(x, gamma, beta, EPS)
    mean = x.mean(-1, keepdim=True)
    c = x - mean
    var = (c * c).mean(-1, keepdim=True)
    n = int(x.shape[-1])
    out = {}
    # tt-train's E[x^2] - E[x]^2 (ops/layernorm_op.cpp:144). Same function in exact arithmetic.
    v2 = (x * x).mean(-1, keepdim=True) - mean * mean
    out["E[x^2]-E[x]^2"] = rel(c * torch.rsqrt(v2 + EPS) * gamma + beta, ref)
    out["unbiased var (N-1)"] = rel(
        c * torch.rsqrt(var * n / (n - 1) + EPS) * gamma + beta, ref)
    out["eps outside the sqrt"] = rel(c / (var.sqrt() + EPS) * gamma + beta, ref)
    out["eps dropped"] = rel(c * torch.rsqrt(var) * gamma + beta, ref)
    out["gamma before normalising"] = rel(
        R.layer_norm_forward(x * gamma, None, beta, EPS), ref)
    out["mean not subtracted (RMS)"] = rel(
        x * torch.rsqrt((x * x).mean(-1, keepdim=True) + EPS) * gamma + beta, ref)
    out["normalised over axis -2"] = rel(
        R.layer_norm_forward(x.transpose(-1, -2), None, None, EPS).transpose(-1, -2)
        * gamma + beta, ref)
    return out


def lin_variants(x, w, bias):
    ref = R.linear_forward(x, w, bias)
    out = {}
    out["bias dropped"] = rel(R.linear_forward(x, w, None), ref)
    out["w transposed"] = rel(R.linear_forward(x, w.transpose(0, 1).contiguous(), bias), ref)
    out["silu fused into the node"] = rel(torch.nn.functional.silu(ref), ref)
    return out


def main() -> int:
    rows = []
    for shape, c in (((1, 64, 64, 128), 128), ((64, 64, 128), 128), ((1, 64, 384), 384)):
        x = torch.randn(*shape, dtype=torch.float64) * 3.0 + 0.7
        gamma = torch.randn(c, dtype=torch.float64)
        beta = torch.randn(c, dtype=torch.float64)
        for nm, v in ln_variants(x, gamma, beta).items():
            rows.append({"op": "layer_norm", "shape": list(shape), "variant": nm, "rel_l2": v})
    for shape, ci, co in (((1, 64, 64, 128), 128, 128), ((64, 64, 128), 128, 128)):
        x = torch.randn(*shape, dtype=torch.float64)
        w = torch.randn(ci, co, dtype=torch.float64) / (ci ** 0.5)
        bias = torch.randn(co, dtype=torch.float64) * 0.1
        for nm, v in lin_variants(x, w, bias).items():
            rows.append({"op": "linear", "shape": list(shape), "variant": nm, "rel_l2": v})

    print("device's own bf16 disagreement with the modelled forward: %.6e" % DEV_BF16)
    print("%-11s %-18s %-28s %12s  %s" % ("op", "shape", "wrong variant", "rel_l2", "separated?"))
    unsep = []
    for r in rows:
        sep = r["rel_l2"] > DEV_BF16
        if not sep:
            unsep.append(r)
        print("%-11s %-18s %-28s %12.4e  %s"
              % (r["op"], str(r["shape"]), r["variant"], r["rel_l2"],
                 "yes" if sep else "NO -- cannot exclude"))
    worst_unsep = max([r["rel_l2"] for r in unsep], default=0.0)
    print()
    print("%d of %d wrong variants read ABOVE the device's own bf16 error and are excluded; "
          "%d read below and are not." % (len(rows) - len(unsep), len(rows), len(unsep)))
    out = {"device_bf16_disagreement": DEV_BF16, "rows": rows,
           "excluded": len(rows) - len(unsep), "not_excluded": len(unsep),
           "worst_not_excluded": worst_unsep,
           "not_excluded_variants": sorted({r["variant"] for r in unsep})}
    if len(sys.argv) > 1:
        Path(sys.argv[1]).write_text(json.dumps(out, indent=1) + "\n")
        print("wrote", sys.argv[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
