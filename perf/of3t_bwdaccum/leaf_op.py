#!/usr/bin/env python3
"""Deliverable 2: the LayerNorm affine backward, in isolation, against float64.

`dW = sum_t g_t * xhat_t` and `db = sum_t g_t` over the leading axes. The device computed both
inside the real backward; this recomputes them in float64 from **the operands the device
actually had** -- the same `x`, the same `g`, the same `gamma`, captured in situ -- so what comes
out is the op's own arithmetic error and not the error that arrived on `g`.

Three numbers decide it and they are reported per site (D35):

  op_rel        the device's dW against the float64 dW from the same operands
  K             the site's cancellation factor, sum_t|term_t| / |sum_t term_t|, per output
                channel. This is what says whether a bf16 summand can throw the sum: a
                correctly-implemented bf16 reduction reads about u*K with u = 2^-8, and a site
                with K = 1 cannot be hurt by one however it is implemented
  bf16_product  the float64 sum of the SAME products rounded to bf16 first. If the device's
                reading sits on this line, the defect is that the summands are built in bf16
                (D56) and not that the reduction accumulates in bf16

`xhat_rel` is beside them, because a wrong xhat is a different defect from a wrong reduction and
the two are not separable from dW alone.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

FOUR = ("pre_norm_s_weight", "pre_norm_s_bias",
        "transition_s.norm_weight", "transition_s.norm_bias")


def triple(m, r):
    m = m.reshape(-1).to(torch.float64)
    r = r.reshape(-1).to(torch.float64)
    nr = float(torch.linalg.vector_norm(r))
    nm = float(torch.linalg.vector_norm(m))
    if nr == 0:
        return None
    return {"rel_l2": float(torch.linalg.vector_norm(m - r) / nr),
            "norm_ratio": nm / nr, "cos": float((m @ r) / (nm * nr)) if nm else 0.0,
            "ref_norm": nr}


def bf16(t):
    return t.to(torch.bfloat16).to(torch.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    U_BF16 = 2.0 ** -8            # bf16 unit roundoff, 8 mantissa bits with the implicit one
    out = {"what": __doc__.strip().splitlines()[0], "bf16_unit_roundoff": U_BF16,
           "arms": {}}
    for spec in a.ln:
        nm, _, p = spec.partition("=")
        d = torch.load(p, map_location="cpu", weights_only=False)
        rows = []
        for site in d["sites"]:
            x = site["x"].to(torch.float64)
            g = site["g"].to(torch.float64)
            eps = float(site["eps"])
            mean = x.mean(dim=-1, keepdim=True)
            centered = x - mean
            var = (centered * centered).mean(dim=-1, keepdim=True)
            rstd = torch.rsqrt(var + eps)
            xhat = centered * rstd
            term = g * xhat
            C = int(xhat.shape[-1])
            flat_t = term.reshape(-1, C)
            dW64 = flat_t.sum(dim=0)
            db64 = g.reshape(-1, C).sum(dim=0)
            # the same sum with the summands rounded to bf16 first, accumulated exactly
            dW_bf16prod = bf16(flat_t).sum(dim=0)
            # and with xhat built the way the device builds it, from bf16 intermediates
            dW_devxhat = (g * site["xhat_device"].to(torch.float64)).reshape(-1, C).sum(dim=0)
            absum = flat_t.abs().sum(dim=0)
            K = (absum / dW64.abs().clamp_min(1e-300))
            w = dW64 * dW64
            w = w / w.sum().clamp_min(1e-300)
            row = {"gamma_path": site["gamma_path"],
                   "block": int(site["gamma_path"].split(".")[1]),
                   "leaf": site["gamma_path"].split(".", 2)[2],
                   "tokens": int(np.prod(list(term.shape[:-1]))), "channels": C,
                   "x_dtype": site["x_dtype"], "g_dtype": site["g_dtype"],
                   "K_median": float(K.median()), "K_max": float(K.max()),
                   "K_mass_weighted": float((w * K).sum()),
                   "predicted_bf16_rel": U_BF16 * float((w * K).sum()),
                   "xhat_rel": triple(site["xhat_device"].to(torch.float64), xhat)["rel_l2"],
                   "dW_bf16product_vs_f64": triple(dW_bf16prod, dW64),
                   "dW_devxhat_vs_f64": triple(dW_devxhat, dW64),
                   "x_norm": float(x.norm()), "g_norm": float(g.norm()),
                   "rstd_median": float(rstd.median())}
            if site["dW_device"] is not None:
                row["dW_device_vs_f64"] = triple(site["dW_device"].to(torch.float64), dW64)
            if site["db_device"] is not None:
                row["db_device_vs_f64"] = triple(site["db_device"].to(torch.float64), db64)
                absum_b = g.reshape(-1, C).abs().sum(dim=0)
                Kb = absum_b / db64.abs().clamp_min(1e-300)
                wb = db64 * db64
                wb = wb / wb.sum().clamp_min(1e-300)
                row["Kb_mass_weighted"] = float((wb * Kb).sum())
                row["db_predicted_bf16_rel"] = U_BF16 * float((wb * Kb).sum())
            rows.append(row)
        rows.sort(key=lambda r: (r["leaf"], r["block"]))
        four = [r for r in rows if r["leaf"] in FOUR]
        other = [r for r in rows if r["leaf"] not in FOUR]
        out["arms"][nm] = {
            "lever": d.get("lever"), "sites": len(rows),
            "blocks_captured": d.get("blocks_captured"),
            "the_four_error_mass_leaves": four,
            "the_other_sites_as_the_passing_contrast": other,
            "summary": {
                "four_dW_rel_median": float(np.median(
                    [r["dW_device_vs_f64"]["rel_l2"] for r in four
                     if r.get("dW_device_vs_f64")]) or 0.0) if four else None,
                "other_dW_rel_median": float(np.median(
                    [r["dW_device_vs_f64"]["rel_l2"] for r in other
                     if r.get("dW_device_vs_f64")])) if other else None,
                "four_K_median": float(np.median([r["K_mass_weighted"] for r in four]))
                if four else None,
                "other_K_median": float(np.median([r["K_mass_weighted"] for r in other]))
                if other else None}}
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)

    for nm, arm in out["arms"].items():
        print(f"=== {nm} (lever {arm['lever']}) ===")
        print(f"{'block':>5} {'leaf':<34} {'K_mw':>10} {'pred':>10} {'dW_dev':>10} "
              f"{'bf16prod':>10} {'devxhat':>10} {'xhat':>10}")
        for r in arm["the_four_error_mass_leaves"] + \
                arm["the_other_sites_as_the_passing_contrast"]:
            dv = r.get("dW_device_vs_f64")
            print(f"{r['block']:>5} {r['leaf']:<34} {r['K_mass_weighted']:10.3e} "
                  f"{r['predicted_bf16_rel']:10.3e} "
                  f"{(dv['rel_l2'] if dv else float('nan')):10.3e} "
                  f"{r['dW_bf16product_vs_f64']['rel_l2']:10.3e} "
                  f"{r['dW_devxhat_vs_f64']['rel_l2']:10.3e} {r['xhat_rel']:10.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
