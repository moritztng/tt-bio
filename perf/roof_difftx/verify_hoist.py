#!/usr/bin/env python3
"""Does the hoisted conditioning half compute the same thing as the shipped one?

Two questions, answered separately because they need different references.

1. THE PRIMITIVES, against float64 and never against another approximation. The hoist changes
   exactly two things arithmetically: the layer_norm gain moves from the activation onto the
   weight, and the transition's output-projection sigmoid moves from its multiply into the
   matmul's own activation epilogue. Both are run ON THE DEVICE in both arrangements and both
   are scored against a float64 host computation of the same quantity.

2. THE MODULE, arm against arm. A real 24-layer `DiffusionTransformer` with random weights, run
   once with the flag off and once with it on. That comparison is approximation against
   approximation, so it is reported as a DISTANCE between the two arms and not as an accuracy
   verdict -- the verdict comes from (1) and from the fold-level structure arm.
"""
from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import torch
import ttnn

import tt_bio.tenstorrent as T

HERE = Path(__file__).resolve().parent
S, DIM, HEADS, NL = 512, 768, 16, 24
INNER = 2 * DIM


def rel_rms(x, ref):
    return float(torch.sqrt(torch.mean((x - ref) ** 2) / torch.mean(ref ** 2)))


def pcc(x, ref):
    a = (x - x.mean()).flatten().double()
    b = (ref - ref.mean()).flatten().double()
    return float((a @ b) / (a.norm() * b.norm()))


def stack_weights(scale=0.02):
    def w(*sh):
        return torch.randn(*sh, dtype=torch.float32) * scale

    d = {}
    for i in range(NL):
        p = f"layers.{i}"
        for ad in (f"{p}.adaln", f"{p}.transition.adaln"):
            d[f"{ad}.s_norm.weight"] = 1.0 + 0.1 * torch.randn(DIM)
            d[f"{ad}.s_scale.weight"] = w(DIM, DIM)
            d[f"{ad}.s_scale.bias"] = w(DIM)
            d[f"{ad}.s_bias.weight"] = w(DIM, DIM)
        for k in ("proj_q", "proj_k", "proj_v", "proj_g", "proj_o"):
            d[f"{p}.pair_bias_attn.{k}.weight"] = w(DIM, DIM)
        d[f"{p}.pair_bias_attn.proj_q.bias"] = w(DIM)
        d[f"{p}.output_projection_linear.weight"] = w(DIM, DIM)
        d[f"{p}.output_projection_linear.bias"] = w(DIM)
        d[f"{p}.transition.swish_gate.0.weight"] = w(2 * INNER, DIM)
        d[f"{p}.transition.a_to_b.weight"] = w(INNER, DIM)
        d[f"{p}.transition.b_to_a.weight"] = w(DIM, INNER)
        d[f"{p}.transition.output_projection.0.weight"] = w(DIM, DIM)
        d[f"{p}.transition.output_projection.0.bias"] = w(DIM)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "verify_hoist.json")
    a = ap.parse_args()
    torch.manual_seed(1)
    torch.set_grad_enabled(False)

    dev = T.get_device()
    kcls = (ttnn.types.WormholeComputeKernelConfig if dev.arch() == ttnn.Arch.WORMHOLE_B0
            else ttnn.types.BlackholeComputeKernelConfig)
    kc = kcls(math_fidelity=ttnn.MathFidelity.HiFi4, math_approx_mode=False,
              fp32_dest_acc_en=True, packer_l1_acc=True)
    out = {"host": platform.node(), "arch": str(dev.arch()),
           "core_grid_main": [T.CORE_GRID_MAIN.x, T.CORE_GRID_MAIN.y]}

    def dev_t(x):
        return ttnn.from_torch(x.to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev,
                               memory_config=ttnn.DRAM_MEMORY_CONFIG)

    # ---- 1a. the gamma fold, on the device, both arrangements, against float64 -----------
    s64 = torch.randn(1, S, DIM, dtype=torch.float64)
    g64 = 1.0 + 0.1 * torch.randn(DIM, dtype=torch.float64)
    w64 = 0.02 * torch.randn(DIM, DIM, dtype=torch.float64)
    sb = s64.to(torch.bfloat16).double()          # what the device is actually given
    mu = sb.mean(-1, keepdim=True)
    var = sb.var(-1, unbiased=False, keepdim=True)
    ref_ad = ((g64 * (sb - mu) / torch.sqrt(var + 1e-5)) @ w64)

    s_d = dev_t(s64.float())
    g_d = ttnn.from_torch(g64.float().to(torch.bfloat16), layout=ttnn.TILE_LAYOUT, device=dev)
    shipped = ttnn.linear(ttnn.layer_norm(s_d, weight=g_d, epsilon=1e-5,
                                          compute_kernel_config=kc),
                          dev_t(w64.float()), compute_kernel_config=kc)
    folded = ttnn.linear(ttnn.layer_norm(s_d, epsilon=1e-5, compute_kernel_config=kc),
                         dev_t((g64.reshape(-1, 1) * w64).float()), compute_kernel_config=kc)
    sh_t, fo_t = ttnn.to_torch(shipped).double(), ttnn.to_torch(folded).double()
    out["gamma_fold"] = {
        "shipped_rel_rms": rel_rms(sh_t, ref_ad), "shipped_pcc": pcc(sh_t, ref_ad),
        "folded_rel_rms": rel_rms(fo_t, ref_ad), "folded_pcc": pcc(fo_t, ref_ad),
        "folded_over_shipped": rel_rms(fo_t, ref_ad) / rel_rms(sh_t, ref_ad),
        "arm_to_arm_rel_rms": rel_rms(fo_t, sh_t)}

    # ---- 1b. where the sigmoid is applied, same treatment --------------------------------
    b64 = 0.02 * torch.randn(DIM, dtype=torch.float64)
    x64 = torch.randn(1, S, DIM, dtype=torch.float64)
    ref_sig = torch.sigmoid(sb @ w64 + b64) * x64.to(torch.bfloat16).double()
    w_d, b_d, x_d = dev_t(w64.float()), dev_t(b64.float().reshape(1, -1)), dev_t(x64.float())
    b_d = ttnn.reshape(b_d, (DIM,))
    at_mul = ttnn.multiply(ttnn.linear(s_d, w_d, bias=b_d, compute_kernel_config=kc), x_d,
                           input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
    in_mm = ttnn.multiply(ttnn.linear(s_d, w_d, bias=b_d, compute_kernel_config=kc,
                                      activation="sigmoid"), x_d)
    am, im = ttnn.to_torch(at_mul).double(), ttnn.to_torch(in_mm).double()
    out["sigmoid_placement"] = {
        "shipped_at_multiply_rel_rms": rel_rms(am, ref_sig),
        "shipped_at_multiply_pcc": pcc(am, ref_sig),
        "hoisted_in_matmul_rel_rms": rel_rms(im, ref_sig),
        "hoisted_in_matmul_pcc": pcc(im, ref_sig),
        "hoisted_over_shipped": rel_rms(im, ref_sig) / rel_rms(am, ref_sig),
        "arm_to_arm_rel_rms": rel_rms(im, am)}

    for t_ in (shipped, folded, at_mul, in_mm):
        ttnn.deallocate(t_)

    # ---- 2. the module, arm against arm --------------------------------------------------
    dt = T.DiffusionTransformer(NL, DIM, HEADS, False, stack_weights(), kc)
    aa = dev_t(torch.randn(1, S, DIM) * 0.5)
    ss = dev_t(torch.randn(1, S, DIM) * 0.5)
    zz = dev_t(torch.randn(1, NL * HEADS, S, S) * 0.1)

    res = {}
    for arm in ("off", "on", "off2"):
        T._B2_DIT_COND_HOIST = arm != "off" and arm != "off2"
        o = dt(aa, ss, zz)
        res[arm] = ttnn.to_torch(o).double()
        ttnn.deallocate(o)
    T._B2_DIT_COND_HOIST = False
    out["module"] = {
        "layers": NL, "tokens": S, "dim": DIM, "heads": HEADS,
        "off_vs_off2_rel_rms": rel_rms(res["off2"], res["off"]),
        "on_vs_off_rel_rms": rel_rms(res["on"], res["off"]),
        "on_vs_off_pcc": pcc(res["on"], res["off"]),
        "on_vs_off_max_abs": float((res["on"] - res["off"]).abs().max()),
        "off_abs_mean": float(res["off"].abs().mean())}

    a.out.write_text(json.dumps(out, indent=1))
    gf, sg, md = out["gamma_fold"], out["sigmoid_placement"], out["module"]
    print("gamma fold vs float64:      shipped %.6e  folded %.6e  -> %.4fx"
          % (gf["shipped_rel_rms"], gf["folded_rel_rms"], gf["folded_over_shipped"]))
    print("sigmoid place vs float64:   at-mul  %.6e  in-mm  %.6e  -> %.4fx"
          % (sg["shipped_at_multiply_rel_rms"], sg["hoisted_in_matmul_rel_rms"],
             sg["hoisted_over_shipped"]))
    print("24-layer module, arm to arm: rel RMS %.6e  PCC %.9f  max abs %.6f (mean |a| %.4f)"
          % (md["on_vs_off_rel_rms"], md["on_vs_off_pcc"], md["on_vs_off_max_abs"],
             md["off_abs_mean"]))
    print("  determinism control, off vs off: rel RMS %.6e" % md["off_vs_off2_rel_rms"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
