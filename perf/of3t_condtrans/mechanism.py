#!/usr/bin/env python3
"""Conditioning, isolation and cotangent substitution for the AdaLN gain, at diffusion scope.

`of3t-lnaffine` located the trunk's equivalent error in the inherited cotangent with three
measurements. This is the same three at this scope, on the device's OWN operands, in float64.

  KAPPA      ||sum_t |g_t * xhat_t||| / ||sum_t g_t * xhat_t||, the cancellation in the sum that
             FORMS this gradient. It bounds the relative error a bf16-class evaluation of the
             contraction can reach at about 2^-9 * KAPPA. A reading above that bound is not
             explained by the leaf's own conditioning.
  ISOLATION  ||dW_device - sum_t g_t * xhat_t|| / ||sum_t g_t * xhat_t||, both sides from the
             SAME operands, so it prices the op's arithmetic and nothing else.
  OPERANDS   ||sum_t g_t * xhat_t - grad_f64|| / ||grad_f64||. The float64 contraction of OUR
             operands against the reference gradient: what is left once the arithmetic is
             removed, and therefore what the operands themselves carry.

Then the substitution, which is teacher-forcing done offline. `layer_norm_s` normalises `s`, the
conditioning, not the per-structure `a`, so `xhat` is the same matrix on every call and the
contraction factorises as sum_rows G * xhat with G the cotangent summed over calls. That is
checked (`xhat_maxdiff`) rather than assumed, and it is what lets the reference cotangent be
substituted into our contraction without a second device run:

  TF-COT     sum_rows(G_ref  * xhat_dev) against grad_f64 -- their cotangent, our xhat
  TF-XHAT    sum_rows(G_dev  * xhat_ref) against grad_f64 -- our cotangent, their xhat
  CONTROL    sum_rows(G_ref  * xhat_ref) against grad_f64 -- must be ~0, or the decomposition is
             not reading the quantity it claims to and nothing above it means anything
"""
from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

LEAF = "conditioned_transition.layer_norm.layer_norm_s.weight"
APB = "attention_pair_bias.layer_norm_a.layer_norm_s.weight"
BF16_ULP = 2.0 ** -9


def rel(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    nb = float(torch.linalg.vector_norm(b))
    return float(torch.linalg.vector_norm(a - b) / nb) if nb else float("nan")


def cos(a, b):
    a = a.reshape(-1).double()
    b = b.reshape(-1).double()
    na, nb = float(a.norm()), float(b.norm())
    return float((a @ b) / (na * nb)) if na and nb else float("nan")


def align(gref, rows):
    """Sum the reference cotangent down to the device's row count, or None if it will not."""
    if gref.shape[0] == rows:
        return gref
    if gref.shape[0] % rows == 0:
        k = gref.shape[0] // rows
        return gref.reshape(k, rows, gref.shape[1]).sum(dim=0)
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-ln", required=True)
    ap.add_argument("--ref-ln", required=True)
    ap.add_argument("--dev-grads", required=True)
    ap.add_argument("--ref-f64", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = torch.load(a.dev_ln, map_location="cpu", weights_only=False)["sites"]
    ref = torch.load(a.ref_ln, map_location="cpu", weights_only=False)["sites"]
    dw = torch.load(a.dev_grads, map_location="cpu", weights_only=False)
    dw = {k[len("diffusion_module."):] if k.startswith("diffusion_module.") else k: v
          for k, v in dw.items()}
    gref = torch.load(a.ref_f64, map_location="cpu", weights_only=False)["grad_f64"]

    rows = {}
    for nm, d in sorted(dev.items()):
        which = LEAF if nm.endswith(LEAF) else (APB if nm.endswith(APB) else None)
        if which is None or nm not in gref or nm not in dw:
            continue
        r = ref.get(nm)
        g64 = gref[nm].reshape(-1).double()
        gsum = d["gsum"]
        kap = float(d["gabs"].norm() / gsum.norm())
        kap0 = (float(d["first_gabs"].norm() / d["first_gsum"].norm())
                if d["first_gsum"] is not None else None)
        e = {"leaf": which, "calls": d["calls"], "rows_dev": d["rows"],
             "KAPPA_all_calls": kap, "KAPPA_first_structure": kap0,
             "bf16_bound_2m9_x_KAPPA": BF16_ULP * kap,
             "ISOLATION": rel(dw[nm].reshape(-1).double(), gsum),
             "OURS_vs_f64": rel(dw[nm].reshape(-1).double(), g64),
             "OPERANDS_vs_f64": rel(gsum, g64),
             "reading_over_bf16_bound": rel(dw[nm].reshape(-1).double(), g64)
                                        / (BF16_ULP * kap) if kap else None,
             "xhat_constant_across_calls_maxdiff": d["xhat_maxdiff"]}
        pc = d.get("per_call") or []
        if r is not None and pc and "G" in r:
            xr_all, Gr_all = r["xhat"], r["G"]
            rpc = pc[0][0].shape[0]
            # The reference ran the 48 structures in one batched call; the device ran 48 tapes.
            # Pairing is by position and it is CHECKED, not assumed: if structure k on the device
            # is not rows [k*rpc, (k+1)*rpc) of the reference, `xhat_pairing_worst` blows up and
            # everything below it is void.
            if xr_all.shape[0] == rpc * len(pc):
                ctrl = torch.zeros(g64.shape[0], dtype=torch.float64)
                tfc = torch.zeros_like(ctrl)
                tfx = torch.zeros_like(ctrl)
                oo = torch.zeros_like(ctrl)
                ce = cr = xe = xrr = dot = nd = nr_ = 0.0
                worst_x = 0.0
                for k, (xd, gd) in enumerate(pc):
                    xrk = xr_all[k * rpc:(k + 1) * rpc]
                    grk = Gr_all[k * rpc:(k + 1) * rpc]
                    ctrl += (grk * xrk).sum(dim=0)
                    tfc += (grk * xd).sum(dim=0)
                    tfx += (gd * xrk).sum(dim=0)
                    oo += (gd * xd).sum(dim=0)
                    ce += float((gd - grk).pow(2).sum())
                    cr += float(grk.pow(2).sum())
                    dot += float((gd * grk).sum())
                    nd += float(gd.pow(2).sum())
                    nr_ += float(grk.pow(2).sum())
                    xe += float((xd - xrk).pow(2).sum())
                    xrr += float(xrk.pow(2).sum())
                    worst_x = max(worst_x, float((xd - xrk).norm() / xrk.norm()))
                e["calls_paired"] = len(pc)
                e["xhat_pairing_worst"] = worst_x
                e["xhat_rel_dev_vs_ref"] = (xe / xrr) ** 0.5
                e["cotangent_rel_dev_vs_ref"] = (ce / cr) ** 0.5
                e["cotangent_cos_dev_vs_ref"] = dot / ((nd ** 0.5) * (nr_ ** 0.5))
                e["cotangent_norm_ratio"] = (nd / nr_) ** 0.5
                e["CONTROL_ref_x_ref"] = rel(ctrl, g64)
                e["TF_COT_ref_cot_our_xhat"] = rel(tfc, g64)
                e["TF_XHAT_our_cot_ref_xhat"] = rel(tfx, g64)
                e["OURS_x_OURS_check"] = rel(oo, g64)
        rows[nm] = e

    def agg(sel, key):
        v = [e[key] for e in rows.values() if e["leaf"] == sel and e.get(key) is not None]
        return {"n": len(v), "median": float(np.median(v)) if v else None,
                "min": float(np.min(v)) if v else None,
                "max": float(np.max(v)) if v else None} if v else None

    keys = ["KAPPA_all_calls", "bf16_bound_2m9_x_KAPPA", "ISOLATION", "OURS_vs_f64",
            "OPERANDS_vs_f64", "reading_over_bf16_bound", "xhat_pairing_worst",
            "xhat_rel_dev_vs_ref", "cotangent_rel_dev_vs_ref", "cotangent_cos_dev_vs_ref",
            "cotangent_norm_ratio", "CONTROL_ref_x_ref", "TF_COT_ref_cot_our_xhat",
            "TF_XHAT_our_cot_ref_xhat", "OURS_x_OURS_check"]
    rep = {"what": __doc__.strip().splitlines()[0],
           "bf16_ulp": BF16_ULP,
           "summary": {lf: {k: agg(lf, k) for k in keys} for lf in (LEAF, APB)},
           "by_site": rows}
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True, default=float)

    for lf in (LEAF, APB):
        print(f"\n=== {lf}")
        print(f"    {'key':<34} {'n':>3} {'median':>11} {'min':>11} {'max':>11}")
        for k in keys:
            s = rep["summary"][lf][k]
            if s:
                print(f"    {k:<34} {s['n']:3d} {s['median']:11.4e} {s['min']:11.4e} "
                      f"{s['max']:11.4e}")
    hdr = ("site", "KAPPA", "bound", "ISO", "OURS", "OPER", "cotrel", "cotcos",
           "TFcot", "TFxhat", "CTRL")
    print("\n%-46s %7s %8s %9s %8s %8s %8s %7s %8s %8s %9s" % hdr)
    for nm, e in sorted(rows.items()):
        if e["leaf"] != LEAF:
            continue

        def num(key, w=8, p=4):
            v = e.get(key)
            return f"{v:{w}.{p}f}" if v is not None else "-".rjust(w)

        def sci(key, w=9, p=2):
            v = e.get(key)
            return f"{v:{w}.{p}e}" if v is not None else "-".rjust(w)

        c = e.get("cotangent_cos_dev_vs_ref")
        cstr = f"{c:+7.3f}" if c is not None else "-".rjust(7)
        print(f"{nm[-46:]:<46} {e['KAPPA_all_calls']:7.2f} "
              f"{e['bf16_bound_2m9_x_KAPPA']:8.4f} {e['ISOLATION']:9.2e} "
              f"{num('OURS_vs_f64')} {num('OPERANDS_vs_f64')} "
              f"{num('cotangent_rel_dev_vs_ref')} {cstr} "
              f"{num('TF_COT_ref_cot_our_xhat')} {num('TF_XHAT_our_cot_ref_xhat')} "
              f"{sci('CONTROL_ref_x_ref')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
