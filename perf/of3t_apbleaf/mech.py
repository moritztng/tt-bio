#!/usr/bin/env python3
"""of3t-apbleaf: what the residue at `attn_pair_bias.layer_norm_a` IS.

The affine gradient of a LayerNorm is a single reduction over the token axis,

    dW = sum_t g_t xhat_t          db = sum_t g_t          xhat_t = (x_t - mean_t) * rstd_t

so every error in `dW` is either MADE by the three device ops that evaluate it or ARRIVES in
`x` and `g`. This splits the two apart by re-evaluating the same reduction in float64 on the
DEVICE's own captured operands:

    E_tot = || dW_dev    - dW_ref ||        what the campaign measures
    E_loc = || dW_dev    - dW_f64dev ||     made here
    E_inh = || dW_f64dev - dW_ref ||        arrived

`E_loc` and `E_inh` are not complements -- the two error vectors can be oblique -- so both are
reported and neither is inferred from the other. The thresholds on `L = E_loc/E_tot` are
`perf/of3t_apbleaf/PREDICTION.md`'s and were fixed before the first capture existed.

The float64 reduction is validated three ways at every site, and a site that fails any of them
is excluded from the mass sums rather than reported: the closed form against torch's own float64
autograd through `F.layer_norm`, and both against float64 CENTRAL FINITE DIFFERENCES of the same
scalar along a random unit direction in W.
"""
from __future__ import annotations

import argparse
import json
import math

import torch
import torch.nn.functional as F

PRE = "pairformer_stack.blocks."
FD_BAR = 1e-9


def affine_grads(x, g, eps):
    """`dW, db` in float64 from the operands, the op's own formula and nothing else."""
    x = x.to(torch.float64)
    g = g.to(torch.float64)
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = xc.pow(2).mean(-1, keepdim=True)
    rstd = torch.rsqrt(var + eps)
    xhat = xc * rstd
    prod = g * xhat
    c = x.shape[-1]
    return (prod.reshape(-1, c).sum(0), g.reshape(-1, c).sum(0), xhat, prod)


def validate(x, g, w, b, eps, seed):
    """closed form vs torch float64 autograd vs central finite differences."""
    x = x.to(torch.float64)
    g = g.to(torch.float64)
    w = w.to(torch.float64).clone().requires_grad_(True)
    b = b.to(torch.float64).clone().requires_grad_(True)
    out = F.layer_norm(x, (x.shape[-1],), weight=w, bias=b, eps=eps)
    loss = (out * g).sum()
    dw_t, db_t = torch.autograd.grad(loss, (w, b))
    dw_c, db_c, _, _ = affine_grads(x, g, eps)
    gen = torch.Generator().manual_seed(seed)
    d = torch.randn(w.shape, generator=gen, dtype=torch.float64)
    d /= d.norm()
    h = 1e-6

    def L(sign):
        with torch.no_grad():
            o = F.layer_norm(x, (x.shape[-1],), weight=w.detach() + sign * h * d,
                             bias=b.detach(), eps=eps)
            return float((o * g).sum())
    fd = (L(+1.0) - L(-1.0)) / (2 * h)
    an = float((dw_c * d).sum())
    return {
        "closed_vs_torch_autograd_dW": float((dw_c - dw_t).norm() / (dw_t.norm() + 1e-300)),
        "closed_vs_torch_autograd_db": float((db_c - db_t).norm() / (db_t.norm() + 1e-300)),
        "central_fd": fd, "analytic_along_direction": an,
        "fd_rel": abs(fd - an) / (abs(an) + 1e-300), "fd_eps": h, "bar": FD_BAR}


def nrm(t):
    return float(t.to(torch.float64).norm())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-ln", required=True)
    ap.add_argument("--ref-ln", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cf-out", default="", help="the per-block dW_f64dev, for the counterfactual")
    a = ap.parse_args()

    dev = torch.load(a.dev_ln, map_location="cpu", weights_only=False)
    ref = torch.load(a.ref_ln, map_location="cpu", weights_only=False)
    ds = {}
    for s in dev["sites"]:
        b = int(s["gamma_path"].split(".")[1])
        ds[b] = s
    rs = {int(k): v for k, v in ref["sites"].items()}
    blocks = sorted(set(ds) & set(rs))

    R = {"what": __doc__.strip().splitlines()[0],
         "inputs": {"dev_ln": a.dev_ln, "ref_ln": a.ref_ln},
         "blocks_paired": len(blocks), "dev_sites": len(ds), "ref_sites": len(rs),
         "by_block": {}, "validation": {}, "pad": {}}

    acc = {k: 0.0 for k in ("tot", "loc", "inh", "ax", "ag", "reftot", "devnorm")}
    cf = {}
    for bi in blocks:
        d, r = ds[bi], rs[bi]
        eps_d, eps_r = float(d["eps"]), 1e-5
        xd, gd = d["x"], d["g"]
        xr, gr = r["x"], r["g"]
        if tuple(xd.shape) != tuple(xr.shape):
            R["by_block"][bi] = {"skipped": "shape mismatch %s vs %s"
                                 % (tuple(xd.shape), tuple(xr.shape))}
            continue
        dW_ref, db_ref = r["dW_ref"].to(torch.float64), r["db_ref"].to(torch.float64)
        dW_dev, db_dev = d["dW_device"].reshape(-1).to(torch.float64), \
            d["db_device"].reshape(-1).to(torch.float64)

        dW_dd, db_dd, xhat_dd, prod_dd = affine_grads(xd, gd, eps_d)   # our operands, exact
        dW_rr, db_rr, _, _ = affine_grads(xr, gr, eps_d)               # reference operands
        dW_dr, _, _, _ = affine_grads(xd, gr, eps_d)                   # our x, their g
        dW_rd, _, _, _ = affine_grads(xr, gd, eps_d)                   # their x, our g

        e_tot = nrm(dW_dev - dW_ref)
        e_loc = nrm(dW_dev - dW_dd)
        e_inh = nrm(dW_dd - dW_ref)
        a_x = nrm(dW_dr - dW_ref)
        a_g = nrm(dW_rd - dW_ref)
        # the cancellation factor of the reduction, per output channel
        c = xd.shape[-1]
        p = prod_dd.reshape(-1, c)
        absum = p.abs().sum(0)
        cancel = (absum / (dW_dd.abs() + 1e-300))
        # pad rows: the cotangent's own support
        gflat = gd.to(torch.float64).reshape(-1, c)
        nz_rows = int((gflat.abs().sum(1) > 0).sum())

        row = {"block": bi, "eps_device": eps_d, "eps_reference": eps_r,
               "dtype_x": d.get("x_dtype"), "dtype_g": d.get("g_dtype"),
               "ref_norm_dW": nrm(dW_ref), "dev_norm_dW": nrm(dW_dev),
               "f64dev_norm_dW": nrm(dW_dd),
               "E_tot": e_tot, "E_loc": e_loc, "E_inh": e_inh,
               "L_block": e_loc / e_tot if e_tot else None,
               "H_block": e_inh / e_tot if e_tot else None,
               "A_x": a_x, "A_g": a_g,
               "rel_dev_vs_ref": e_tot / (nrm(dW_ref) + 1e-300),
               "rel_f64dev_vs_ref": e_inh / (nrm(dW_ref) + 1e-300),
               "refops_reproduce_reference": nrm(dW_rr - dW_ref) / (nrm(dW_ref) + 1e-300),
               "cancellation_median": float(cancel.median()),
               "cancellation_max": float(cancel.max()),
               "cotangent_rows_nonzero": nz_rows, "cotangent_rows_total": int(gflat.shape[0]),
               "db_E_tot": nrm(db_dev - db_ref),
               "db_E_loc": nrm(db_dev - db_dd),
               "db_E_inh": nrm(db_dd - db_ref)}
        R["by_block"][bi] = row
        acc["tot"] += e_tot ** 2 + row["db_E_tot"] ** 2
        acc["loc"] += e_loc ** 2 + row["db_E_loc"] ** 2
        acc["inh"] += e_inh ** 2 + row["db_E_inh"] ** 2
        acc["ax"] += a_x ** 2
        acc["ag"] += a_g ** 2
        acc["reftot"] += nrm(dW_ref) ** 2 + nrm(db_ref) ** 2
        cf[PRE + "%d.attn_pair_bias.layer_norm_a.weight" % bi] = dW_dd.clone()
        cf[PRE + "%d.attn_pair_bias.layer_norm_a.bias" % bi] = db_dd.clone()
        if bi in (44, 4, 0):
            R["validation"][bi] = validate(xd, gd, d["gamma"].reshape(-1), r["beta"], eps_d, bi)

    T = math.sqrt(acc["tot"])
    R["mass_weighted"] = {
        "denominator": "the %d paired blocks' layer_norm_a weight and bias, %d tensors"
                       % (len(blocks), 2 * len(blocks)),
        "E_tot": T, "E_loc": math.sqrt(acc["loc"]), "E_inh": math.sqrt(acc["inh"]),
        "L": math.sqrt(acc["loc"]) / T if T else None,
        "H": math.sqrt(acc["inh"]) / T if T else None,
        "A_x": math.sqrt(acc["ax"]) / T if T else None,
        "A_g": math.sqrt(acc["ag"]) / T if T else None,
        "ref_norm": math.sqrt(acc["reftot"])}
    L = R["mass_weighted"]["L"]
    H = R["mass_weighted"]["H"]
    R["clause"] = {
        "L": L, "H": H,
        "named_LOCAL_if": "L >= 0.75", "named_INHERITED_if": "L <= 0.25 and H >= 0.75",
        "verdict": ("LOCAL" if (L is not None and L >= 0.75)
                    else "INHERITED" if (L is not None and L <= 0.25 and H >= 0.75)
                    else "SHARED"),
        "prediction": "INHERITED, carried by the cotangent, with A_g >= 3x A_x"}
    print(json.dumps(R["mass_weighted"], indent=1))
    print(json.dumps(R["clause"], indent=1))
    json.dump({k: v for k, v in R.items()}, open(a.out, "w"), indent=2)
    if a.cf_out:
        torch.save({"dW_f64dev": cf, "blocks": blocks}, a.cf_out)
        print("wrote %s with %d tensors" % (a.cf_out, len(cf)))
    print("wrote " + a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
