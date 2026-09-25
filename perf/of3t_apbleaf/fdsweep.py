#!/usr/bin/env python3
"""of3t-apbleaf: the finite-difference clause, re-read across step sizes.

`PREDICTION.md` fixed the validation bar at `rel <= 1e-9` for float64 central differences, and
at h = 1e-6 two of the three checked sites came in ABOVE it: 5.455e-09 at block 0 and 1.666e-09
at block 4. The bar is not moved. What is reported instead is where the miss comes from, and it
is the instrument: `L(W) = <g, xhat W + beta>` is AFFINE in W, so central differences carry no
truncation error at all and the entire residual is the cancellation in `L(+h) - L(-h)`, whose
relative size is about `eps * |L(0)| / (h * |<dW, d>|)`. That predicts the residual falls like
1/h, which a sweep either shows or does not.

The independent check that does not go through a subtraction is in `mech.py`: the closed form
against torch's own float64 autograd through `F.layer_norm`, 3.8e-16, 7.9e-16 and 6.5e-17 at the
same three sites.
"""
import argparse, json
import torch
import torch.nn.functional as F


def affine_dW(x, g, eps):
    x = x.to(torch.float64); g = g.to(torch.float64)
    mu = x.mean(-1, keepdim=True); xc = x - mu
    var = xc.pow(2).mean(-1, keepdim=True)
    xhat = xc * torch.rsqrt(var + eps)
    c = x.shape[-1]
    return (g * xhat).reshape(-1, c).sum(0)


ap = argparse.ArgumentParser()
ap.add_argument("--dev-ln", required=True)
ap.add_argument("--ref-ln", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
dev = torch.load(a.dev_ln, map_location="cpu", weights_only=False)
ref = torch.load(a.ref_ln, map_location="cpu", weights_only=False)
ds = {int(s["gamma_path"].split(".")[1]): s for s in dev["sites"]}
rs = {int(k): v for k, v in ref["sites"].items()}
R = {"what": __doc__.strip().splitlines()[0], "bar": 1e-9, "sites": {}}
for b in (0, 4, 44):
    d, r = ds[b], rs[b]
    x = d["x"].to(torch.float64); g = d["g"].to(torch.float64)
    w = d["gamma"].reshape(-1).to(torch.float64)
    beta = r["beta"].to(torch.float64)
    eps = float(d["eps"])
    dw = affine_dW(x, g, eps)
    gen = torch.Generator().manual_seed(b)
    dr = torch.randn(w.shape, generator=gen, dtype=torch.float64); dr /= dr.norm()
    an = float((dw * dr).sum())
    L0 = float((F.layer_norm(x, (x.shape[-1],), weight=w, bias=beta, eps=eps) * g).sum())
    row = {"analytic_along_direction": an, "L_at_zero": L0,
           "predicted_cancellation_rel_at_h": {}, "sweep": {}}
    for h in (1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        fp = float((F.layer_norm(x, (x.shape[-1],), weight=w + h * dr, bias=beta, eps=eps) * g).sum())
        fm = float((F.layer_norm(x, (x.shape[-1],), weight=w - h * dr, bias=beta, eps=eps) * g).sum())
        fd = (fp - fm) / (2 * h)
        row["sweep"]["%g" % h] = {"fd": fd, "rel": abs(fd - an) / (abs(an) + 1e-300)}
        row["predicted_cancellation_rel_at_h"]["%g" % h] = \
            2.220446049250313e-16 * abs(L0) / (h * abs(an) + 1e-300)
    R["sites"][b] = row
    print("block %2d  analytic=%.10e  L(0)=%.6e" % (b, an, L0))
    for h, v in row["sweep"].items():
        print("   h=%-6s fd=%.10e  rel=%.3e   predicted cancellation %.3e"
              % (h, v["fd"], v["rel"], row["predicted_cancellation_rel_at_h"][h]))
json.dump(R, open(a.out, "w"), indent=2)
print("wrote " + a.out)
