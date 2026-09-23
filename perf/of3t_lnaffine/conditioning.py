#!/usr/bin/env python3
"""How ill-conditioned is `g_gamma = sum_t g_t * xhat_t`, per LayerNorm site, measured.

D56 interpolated a cancellation condition number of 1.3e+06 for this contraction and D62
measured 172.60 at one site. Neither was measured at the site D8 names. This does it from the
operands the DEVICE actually held, captured in situ by `dev_cot.py --ln-capture`, with the
contraction itself evaluated in float64 so the conditioning is separated from the arithmetic.

Per site, over the token axis:

    kappa_c  = sum_t |g_tc * xhat_tc|  /  |sum_t g_tc * xhat_tc|          per channel c
    KAPPA    = || sum_t |g_t * xhat_t| ||  /  || sum_t g_t * xhat_t ||    one number per site

KAPPA is the amplification a relative rounding error on the summands gets in the result. It is
1 when every summand has the same sign and unbounded as the sum cancels. A bf16 summand carries
8 mantissa bits, so the error floor any bf16-class evaluation of this sum can reach is about
`2^-9 * KAPPA` in relative terms -- and that bound applies to upstream's own bf16 arm exactly as
it applies to ours.

ISOLATION is reported beside it: the device's own dW against the float64 contraction of the
device's OWN operands. That separates "this op's arithmetic is wrong" from "this quantity
cannot be computed accurately from these inputs".
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

# their name <- our name. `attn_pair_bias.layer_norm_a` is the PairformerLayer's own s
# LayerNorm, applied outside the module: perf/of3t_bwdaccum/ref_cot.py:126 and
# perf/of3t_confidence/grad_device.py:101 both record the pairing.
THEIRS = {
    "pre_norm_s": "attn_pair_bias.layer_norm_a",
    "attention_pair_bias.z_norm": "attn_pair_bias.layer_norm_z",
    "transition_s.norm": "single_transition.layer_norm",
    "transition_z.norm": "pair_stack.pair_transition.layer_norm",
    "triangle_attention_start.layer_norm": "pair_stack.tri_att_start.layer_norm",
    "triangle_attention_end.layer_norm": "pair_stack.tri_att_end.layer_norm",
    "triangle_multiplication_start.in_norm": "pair_stack.tri_mul_in.layer_norm_in",
    "triangle_multiplication_end.in_norm": "pair_stack.tri_mul_out.layer_norm_in",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ln", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    cap = torch.load(a.ln, map_location="cpu", weights_only=False)
    rep = {"what": __doc__.strip().splitlines()[0], "ln": a.ln,
           "lever": cap["lever"], "blocks": cap["blocks_captured"], "sites": {}}

    for s in cap["sites"]:
        path = s["gamma_path"]                       # blocks.<i>.<sub>_weight
        blk = int(path.split(".")[1])
        sub = path[len(f"blocks.{blk}."):]
        sub = sub[:-len("_weight")] if sub.endswith("_weight") else sub
        theirs = THEIRS.get(sub, "?" + sub)

        x = s["x"].to(torch.float64)
        g = s["g"].to(torch.float64)
        eps = s["eps"]
        mean = x.mean(-1, keepdim=True)
        cen = x - mean
        rstd = torch.rsqrt((cen * cen).mean(-1, keepdim=True) + eps)
        xhat = cen * rstd

        C = x.shape[-1]
        prod = (g * xhat).reshape(-1, C)             # the summands, one row per token slot
        exact = prod.sum(0)                          # float64 g_gamma from the device's operands
        absum = prod.abs().sum(0)

        kap_c = (absum / exact.abs().clamp_min(1e-300)).numpy()
        KAPPA = float(absum.norm() / exact.norm()) if float(exact.norm()) else float("inf")

        row = {"their_name": theirs, "block": blk, "summands_per_channel": prod.shape[0],
               "channels": C,
               "KAPPA": KAPPA,
               "kappa_per_channel_median": float(np.median(kap_c)),
               "kappa_per_channel_p90": float(np.percentile(kap_c, 90)),
               "kappa_per_channel_max": float(kap_c.max()),
               "bf16_reachable_rel_floor_2pow_minus9_times_KAPPA": (2 ** -9) * KAPPA,
               "exact_norm_from_device_operands": float(exact.norm())}

        dw = s.get("dW_device")
        if dw is not None:
            d = dw.reshape(-1).to(torch.float64)
            e = exact.reshape(-1)
            row["ISOLATION_rel_device_vs_f64_on_own_operands"] = float((d - e).norm() / e.norm())
            row["ISOLATION_cos"] = float((d @ e) / (d.norm() * e.norm()))
        rep["sites"][f"{blk}/{theirs}"] = row

    # group by leaf so the answer is per leaf, which is how D8 states the question
    grp = {}
    for k, v in rep["sites"].items():
        grp.setdefault(v["their_name"], []).append(v)
    rep["by_leaf"] = {
        k: {"n": len(v),
            "KAPPA_median": float(np.median([r["KAPPA"] for r in v])),
            "KAPPA_min": min(r["KAPPA"] for r in v),
            "KAPPA_max": max(r["KAPPA"] for r in v),
            "bf16_reachable_rel_floor_median":
                float(np.median([r["bf16_reachable_rel_floor_2pow_minus9_times_KAPPA"]
                                 for r in v])),
            "ISOLATION_rel_max":
                max(r.get("ISOLATION_rel_device_vs_f64_on_own_operands", 0.0) for r in v),
            "blocks": sorted(r["block"] for r in v)}
        for k, v in sorted(grp.items(), key=lambda kv: -np.median([r["KAPPA"] for r in kv[1]]))}

    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1, sort_keys=True)
    print(f"{'leaf':46s} {'n':>2} {'KAPPA med':>10} {'min':>9} {'max':>10} "
          f"{'bf16 floor':>11} {'ISOL max':>9}")
    for k, v in rep["by_leaf"].items():
        print(f"{k:46s} {v['n']:2d} {v['KAPPA_median']:10.2f} {v['KAPPA_min']:9.2f} "
              f"{v['KAPPA_max']:10.2f} {v['bf16_reachable_rel_floor_median']:11.4f} "
              f"{v['ISOLATION_rel_max']:9.2e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
