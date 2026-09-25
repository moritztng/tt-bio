#!/usr/bin/env python3
"""of3t-apbleaf: the cotangent that arrives at `attn_pair_bias.layer_norm_a`, scored directly.

`MECH_N384.json` puts 99.70 % of the affine gradient's error on `g` and 0.65 % on `x`, so this
scores `g` itself against the float64 reference and against UPSTREAM'S OWN bf16 `g` at the same
site. Two things come out of it that the dW figure cannot say:

  - the AMPLIFICATION, rel(dW) / rel(g). The reduction sum_t g_t xhat_t cancels, so a small
    relative error in `g` arrives at `dW` multiplied. Naming that factor is what separates "the
    cotangent is wrong" from "the cotangent is ordinary and the reduction is ill-conditioned".
  - whether the residue over upstream is IN `g`. If our `g` and upstream's bf16 `g` carry the
    same relative error then the 2.4044x mass-weighted residue at dW is not made in the single
    track's cotangent at all, and the next row looks elsewhere.

Everything is masked two ways: over all 384 padded rows and over the 56 rows the cotangent is
non-zero on, because 85.4 % of the reduction is structurally zero and a figure over the padded
width understates the real error per contributing row.
"""
from __future__ import annotations

import argparse, json, math
import torch


def rel(a, b):
    a = a.to(torch.float64).reshape(-1); b = b.to(torch.float64).reshape(-1)
    d = float((a - b).norm()); n = float(b.norm())
    return {"abs_err": d, "ref_norm": n, "rel_l2": d / n if n else None,
            "norm_ratio": float(a.norm()) / n if n else None,
            "cos": float((a * b).sum() / (a.norm() * b.norm() + 1e-300))}


ap = argparse.ArgumentParser()
ap.add_argument("--dev-ln", required=True)
ap.add_argument("--ref-ln", required=True)
ap.add_argument("--bf16-ln", default="")
ap.add_argument("--mech", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

dev = torch.load(a.dev_ln, map_location="cpu", weights_only=False)
ref = torch.load(a.ref_ln, map_location="cpu", weights_only=False)
M = json.load(open(a.mech))["by_block"]
bf = (torch.load(a.bf16_ln, map_location="cpu", weights_only=False)["sites"]
      if a.bf16_ln else None)
ds = {int(s["gamma_path"].split(".")[1]): s for s in dev["sites"]}
rs = {int(k): v for k, v in ref["sites"].items()}

R = {"what": __doc__.strip().splitlines()[0],
     "inputs": {"dev": a.dev_ln, "ref": a.ref_ln, "bf16": a.bf16_ln}, "by_block": {}}
acc = {k: 0.0 for k in ("gd", "gb", "gref", "xd", "xref")}
for b in sorted(set(ds) & set(rs)):
    gd, gr = ds[b]["g"], rs[b]["g"]
    xd, xr = ds[b]["x"], rs[b]["x"]
    c = gr.shape[-1]
    m = (gr.to(torch.float64).reshape(-1, c).abs().sum(1) > 0)
    row = {"block": b,
           "g_dev_vs_ref": rel(gd, gr),
           "g_dev_vs_ref_realrows": rel(gd.to(torch.float64).reshape(-1, c)[m],
                                        gr.to(torch.float64).reshape(-1, c)[m]),
           "x_dev_vs_ref": rel(xd, xr),
           "x_dev_vs_ref_realrows": rel(xd.to(torch.float64).reshape(-1, c)[m],
                                        xr.to(torch.float64).reshape(-1, c)[m]),
           "real_rows": int(m.sum()), "rows": int(m.numel())}
    if bf is not None and b in {int(k) for k in bf}:
        gb = bf[b]["g"] if b in bf else bf[str(b)]["g"]
        xb = bf[b]["x"] if b in bf else bf[str(b)]["x"]
        row["g_bf16_vs_ref"] = rel(gb, gr)
        row["g_bf16_vs_ref_realrows"] = rel(gb.to(torch.float64).reshape(-1, c)[m],
                                            gr.to(torch.float64).reshape(-1, c)[m])
        row["x_bf16_vs_ref"] = rel(xb, xr)
        rg = row["g_dev_vs_ref"]["rel_l2"]; rb = row["g_bf16_vs_ref"]["rel_l2"]
        row["g_residue_factor"] = rg / rb if rb else None
        acc["gb"] += row["g_bf16_vs_ref"]["abs_err"] ** 2
    mb = M[str(b)]
    row["dW_rel"] = mb["rel_dev_vs_ref"]
    row["amplification_dW_over_g"] = (mb["rel_dev_vs_ref"] / row["g_dev_vs_ref"]["rel_l2"]
                                      if row["g_dev_vs_ref"]["rel_l2"] else None)
    row["cancellation_median"] = mb["cancellation_median"]
    R["by_block"][b] = row
    acc["gd"] += row["g_dev_vs_ref"]["abs_err"] ** 2
    acc["gref"] += row["g_dev_vs_ref"]["ref_norm"] ** 2
    acc["xd"] += row["x_dev_vs_ref"]["abs_err"] ** 2
    acc["xref"] += row["x_dev_vs_ref"]["ref_norm"] ** 2

R["mass_weighted"] = {
    "g_dev_vs_ref_rel_l2": math.sqrt(acc["gd"] / acc["gref"]),
    "g_bf16_vs_ref_rel_l2": (math.sqrt(acc["gb"] / acc["gref"]) if acc["gb"] else None),
    "g_residue_factor": (math.sqrt(acc["gd"] / acc["gb"]) if acc["gb"] else None),
    "x_dev_vs_ref_rel_l2": math.sqrt(acc["xd"] / acc["xref"]),
    "denominator": "the 48 captured sites' float64 cotangent norm, summed in squares"}
print(json.dumps(R["mass_weighted"], indent=1))
print("\n  blk  rel(g)dev  rel(g)bf16  g_resid   rel(x)dev   rel(dW)   amp=relW/relg  cancel_med")
for b, r in sorted(R["by_block"].items(), key=lambda kv: -kv[1]["dW_rel"])[:12]:
    print("  %3d  %.3e  %-10s %-9s %.3e  %8.3f  %10.1f  %8.1f"
          % (b, r["g_dev_vs_ref"]["rel_l2"],
             ("%.3e" % r["g_bf16_vs_ref"]["rel_l2"]) if "g_bf16_vs_ref" in r else "-",
             ("%.3fx" % r["g_residue_factor"]) if r.get("g_residue_factor") else "-",
             r["x_dev_vs_ref"]["rel_l2"], r["dW_rel"],
             r["amplification_dW_over_g"], r["cancellation_median"]))
print("\nnamed blocks:")
for b in (44, 4, 0, 42):
    r = R["by_block"][b]
    print("  blk %-3d rel(g)=%.4e cos=%+.6f ratio=%.6f | realrows rel(g)=%.4e | rel(x)=%.4e | amp=%.1f"
          % (b, r["g_dev_vs_ref"]["rel_l2"], r["g_dev_vs_ref"]["cos"],
             r["g_dev_vs_ref"]["norm_ratio"], r["g_dev_vs_ref_realrows"]["rel_l2"],
             r["x_dev_vs_ref"]["rel_l2"], r["amplification_dW_over_g"]))
    if "g_bf16_vs_ref" in r:
        print("          upstream bf16 rel(g)=%.4e cos=%+.6f  residue %.4fx"
              % (r["g_bf16_vs_ref"]["rel_l2"], r["g_bf16_vs_ref"]["cos"], r["g_residue_factor"]))
json.dump(R, open(a.out, "w"), indent=2)
print("wrote " + a.out)
