#!/usr/bin/env python3
"""A18 re-asserted in this row's own process, and the gradient's depth profile.

A18's first clause gates a gradient on the forward at its own boundary, and a forward carried in
from another row's write-up is not that. Both device arms wrote their own forward tensors beside
their gradients in the same process; this scores them against the same 0.4.3 float64 reference
the gradient is scored against, masked to the 56 real tokens (D95: mask, always).

The depth profile is the free half. The error mass concentrates in the blocks the backward
reaches LAST, which is the opposite end of the stack from the forward's cliff, so the per-block
series is printed in block order rather than sorted -- a backward that accumulates and a
backward with a localised defect look different in it.
"""
import json
import sys

import numpy as np
import torch

G = "/home/ttuser/of3t_trunkg043/"
B = "/home/ttuser/of3t_trunk043ref/boundary_c64.pt"


def L(f):
    return torch.load(G + f, map_location="cpu", weights_only=False)


def rel(a, b):
    return float(torch.linalg.vector_norm(a - b) / (torch.linalg.vector_norm(b) + 1e-300))


def main():
    b = torch.load(B, map_location="cpu", weights_only=False)
    sm = b["single_mask"].reshape(-1) > 0
    N = int(sm.numel())
    msk = sm.to(torch.float64).reshape(1, N, 1)
    # THEIR pair mask as captured, not an outer product reconstructed from the single mask.
    # The two agree here, and the check says so rather than assuming it.
    pmk = b["pair_mask"].to(torch.float64).reshape(1, N, N, 1)
    outer = (sm.to(torch.float64).reshape(1, N, 1) * sm.to(torch.float64).reshape(1, 1, N)
             ).reshape(1, N, N, 1)
    ref = L("ref_f64_c64.pt")
    bf = L("ref_bf16auto_c64.pt")
    rs = ref["s"].reshape(1, N, -1).to(torch.float64)
    rz = ref["z"].reshape(1, N, N, -1).to(torch.float64)
    out = {"what": __doc__.strip().splitlines()[0],
           "reference": "upstream 0.4.3 float64 over the same boundary, this row's own run",
           "masking": f"{int(sm.sum())} real of {N} tokens; single padding fraction "
                      f"{1 - int(sm.sum()) / N:.6f}, pair {1 - (int(sm.sum()) / N) ** 2:.6f}",
           "pair_mask_is_the_outer_product_of_the_single_mask": bool(torch.equal(pmk, outer)),
           "A18_bar_per_tensor": 5.0e-02, "A18_reachable_sqrt2": 5.0e-02 * 2 ** 0.5, "arms": {}}
    arms = {"FLIPPED": "dev_flipped_c64.pt", "SHIPPED": "dev_shipped_c64.pt",
            "UPSTREAM_BF16": None}
    for name, f in arms.items():
        d = bf if f is None else L(f)
        s = d["s"].reshape(1, N, -1).to(torch.float64)
        z = d["z"].reshape(1, N, N, -1).to(torch.float64)
        out["arms"][name] = {
            "s_masked": rel(s * msk, rs * msk), "z_masked": rel(z * pmk, rz * pmk),
            "s_unmasked": rel(s, rs), "z_unmasked": rel(z, rz),
            "s_norm_masked": float((s * msk).norm()), "z_norm_masked": float((z * pmk).norm()),
            "config": d.get("config", "upstream fp32 params under autocast('cpu', bfloat16)")}
    # the pair track under the flip must be bit-identical: a flag that reaches one of the
    # layer's two kernels cannot move the other
    zf = L("dev_flipped_c64.pt")["z"].to(torch.float64)
    zs = L("dev_shipped_c64.pt")["z"].to(torch.float64)
    out["pair_track_bit_identical_under_the_flip"] = {
        "torch_equal": bool(torch.equal(zf, zs)),
        "z_norm_flipped": float(zf.norm()), "z_norm_shipped": float(zs.norm())}

    sc = json.load(open("perf/of3t_trunkg043/SCORE_c64.json"))
    out["depth_profile"] = {
        "in_block_order": True,
        "note": "mass-weighted rel_l2 within each block, beside that block's share of the "
                "stack's squared gradient norm. Block 47 is the first the backward reaches.",
        "blocks": {i: {"mass_share": sc["arms"]["FLIPPED"]["per_block"][str(i)][
                           "mass_share_of_the_stack"],
                       "FLIPPED": sc["arms"]["FLIPPED"]["per_block"][str(i)][
                           "mass_weighted_rel_l2"],
                       "SHIPPED": sc["arms"]["SHIPPED"]["per_block"][str(i)][
                           "mass_weighted_rel_l2"],
                       "FLOOR": sc["floor_per_block"][str(i)]["mass_weighted_rel_l2"],
                       "FLIPPED_over_FLOOR": (
                           sc["arms"]["FLIPPED"]["per_block"][str(i)]["mass_weighted_rel_l2"]
                           / sc["floor_per_block"][str(i)]["mass_weighted_rel_l2"]),
                       "SHIPPED_over_FLOOR": (
                           sc["arms"]["SHIPPED"]["per_block"][str(i)]["mass_weighted_rel_l2"]
                           / sc["floor_per_block"][str(i)]["mass_weighted_rel_l2"])}
                   for i in range(48)}}
    # the cotangent's own journey: ds_in / dz_in at block 0's input, ours against theirs
    refrep = json.load(open("perf/of3t_trunkg043/REF_F64_c64.json"))
    out["input_cotangent_at_block_0"] = {"reference": {
        "ds_in_norm": refrep["ds_in_norm"], "dz_in_norm": refrep["dz_in_norm"]}}
    for name, f in (("FLIPPED", "DEV_FLIPPED_c64.json"), ("SHIPPED", "DEV_SHIPPED_c64.json")):
        r = json.load(open("perf/of3t_trunkg043/" + f))
        g = r["input_grad_ours"]
        out["input_cotangent_at_block_0"][name] = {
            "ds_in_norm": g["ds_in_norm"], "dz_in_norm": g["dz_in_norm"],
            "ds_ratio": (g["ds_in_norm"] / refrep["ds_in_norm"]) if refrep["ds_in_norm"] else None,
            "dz_ratio": (g["dz_in_norm"] / refrep["dz_in_norm"]) if refrep["dz_in_norm"] else None}

    json.dump(out, open(sys.argv[1], "w"), indent=2)
    for n, v in out["arms"].items():
        print(f"{n:14s} s_masked {v['s_masked']:.6e}  z_masked {v['z_masked']:.6e}")
    print("pair track bit-identical under the flip:",
          out["pair_track_bit_identical_under_the_flip"]["torch_equal"])
    dp = out["depth_profile"]["blocks"]
    print("depth (block: mass%, FLIPPED, FLOOR)")
    for i in list(range(0, 48, 4)) + [44, 45, 46, 47]:
        print("  %2d  %6.2f%%  %10.3e  %10.3e  x%.1f" % (
            i, 100 * dp[i]["mass_share"], dp[i]["FLIPPED"], dp[i]["FLOOR"],
            dp[i]["FLIPPED_over_FLOOR"]))
    print("input cotangent:", json.dumps(out["input_cotangent_at_block_0"], indent=1))


main()
