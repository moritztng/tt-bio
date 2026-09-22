#!/usr/bin/env python3
"""The AdaLN micro-arm: one sub-module, isolated, against a float64 reference.

Row `of3t-adaln`. The campaign's worst gradient disagreement is rel 18.504 on
`diffusion_module.diffusion_transformer.blocks.8.attention_pair_bias.layer_norm_a.layer_norm_s.weight`,
8.05416 % of OpenFold3's squared gradient norm and its fourth-heaviest tensor. Every one of the
ten worst tensors in the diffusion device arm is a LayerNorm gain or a `linear_g` -- the
sigmoid-gated branch of AdaLN -- and not one is a `linear_s`. Four candidate causes were
eliminated by source read, which is an argument, not a measurement. This is the measurement.

ONE `tenstorrent.AdaLN` built from one DiT block's real checkpoint weights, run taped on
real-shaped `(a, s)`, seeded with a fixed cotangent at the output, compared against a float64
torch autograd reference built from upstream 0.4.3's OWN `AdaLN` class
(`openfold3/core/model/primitives/normalization.py:88`) with the same weights, the same inputs
and the same cotangent.

Per tensor it reports `rel_l2`, the norm ratio `r = ||g_dev||/||g_ref||` AND the cosine. rel
alone bounds r to [1-rel, 1+rel] and cannot tell a 19.5x magnitude error from an orthogonal
direction; those have completely different causes.

Arms (`--sweep` runs the whole matrix in one device context):

  gate      `fused` is the shipped path, `multiply_(a, s_scale,
            input_tensor_b_activations=[SIGMOID])` then `add_`. `unfused` repeats the module's
            body with `sigmoid` as its own taped op and a plain out-of-place `multiply`, so a
            disagreement between the two localises to those two lines and nothing else.
  s-scale   PROTOCOL A22: dirty mantissas (0.1, 0.0137, 3.7), never powers of two, which change
            no rounding decision and make a scale arm incapable of failing.
  cot-scale the arithmetic control. The backward is EXACTLY linear in the cotangent, so a
            power-of-two cotangent scale MUST leave rel, r and cos bit-identical. Unlike an
            input scale -- layer_norm's eps and the sigmoid are not homogeneous in `s`, so a
            power-of-two `s` arm is only nearly invariant -- this one is a control that can
            actually fail.
  samples   N tapes accumulating into the same four leaves, the way the diffusion arm sums its
            48 noise levels.
  cot-corr  the cancellation probe. A LayerNorm gain gradient is a SUM over tokens, and its
            relative error is set by the size of the TERMS against the size of the SUM. rho is
            how correlated the seeded cotangent is across tokens; the arm reports the measured
            cancellation ratio K = sum_i ||term_i|| / ||sum_i term_i|| beside each reading, so
            rel can be read against how much cancellation produced it.

A16: the measured zero-model baseline is emitted beside every reading. A zero gradient reads
rel = 1 exactly, and without the baseline you cannot tell "wrong" from "absent".

Nothing in `tt_bio/` is touched. This is an instrument.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
OUT = "perf/of3t_adaln"
C_A, C_S = 768, 384
# Measured on qb2 from bundle_min_043/grads_f64_043.pt, 4,170 tensors. It is the model's OWN
# squared gradient norm, not this arm's.
MODEL_SQ_NORM = 10.279642678524985
# Published reference mass of the four parameters this arm compares, per block. Context for the
# reading, never a denominator for it: this arm feeds synthetic inputs, so its own reference
# gradient is a different vector from the published one.
PUBLISHED = {
    0:  {"layer_norm_s.weight": 0.18134, "linear_g.bias": 0.00089,
         "linear_g.weight": 0.00013, "linear_s.weight": 0.00643},
    5:  {"layer_norm_s.weight": 2.43308, "linear_g.bias": 0.01425,
         "linear_g.weight": 0.00396, "linear_s.weight": 0.10864},
    6:  {"layer_norm_s.weight": 1.39118, "linear_g.bias": 0.00136,
         "linear_g.weight": 0.00011, "linear_s.weight": 0.01139},
    7:  {"layer_norm_s.weight": 2.86053, "linear_g.bias": 0.01424,
         "linear_g.weight": 0.00227, "linear_s.weight": 0.04503},
    8:  {"layer_norm_s.weight": 8.05416, "linear_g.bias": 0.02472,
         "linear_g.weight": 0.00413, "linear_s.weight": 0.05522},
    12: {"layer_norm_s.weight": 0.76487, "linear_g.bias": 0.00229,
         "linear_g.weight": 0.00003, "linear_s.weight": 0.00784},
}
PARAMS = ("layer_norm_s.weight", "linear_g.weight", "linear_g.bias", "linear_s.weight")
PER_TENSOR_BAR = 5.0e-02
POW2 = {2.0 ** k for k in range(-12, 13)}


def _rel(x, y):
    import torch
    x, y = x.reshape(-1).double(), y.reshape(-1).double()
    return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--block", type=int, default=8)
    p.add_argument("--gate", default="fused", choices=["fused", "unfused"])
    p.add_argument("--act", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--tokens", type=int, default=384)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--s-scale", type=float, default=1.0, dest="s_scale")
    p.add_argument("--a-scale", type=float, default=1.0, dest="a_scale")
    p.add_argument("--cot-scale", type=float, default=1.0, dest="cot_scale")
    p.add_argument("--cot-corr", type=float, default=0.0, dest="cot_corr")
    p.add_argument("--ln-rule", default="shipped", dest="ln_rule",
                   choices=["shipped", "fp32_summand", "worse_summand"],
                   help="amendment 4: how the layer_norm gain gradients summand product is "
                        "formed before `_sum_leading` sums it precisely")
    p.add_argument("--fp32-weights", action="store_true", dest="fp32_weights",
                   help="replace AdaLN's four bf16 device weights with fp32 copies, outside "
                        "tt_bio/, to price what the hardcoded bf16 costs the gradient")
    p.add_argument("--cancel", type=float, default=None,
                   help="the cancellation ladder: constant (a, s) across tokens and a "
                        "sign-alternating cotangent, so the gain gradient's 384 summands cancel "
                        "down to this residual and K rises as 1/cancel")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--block-ladder", action="store_true", dest="block_ladder",
                   help="the same synthetic inputs through every block that appears in the "
                        "campaign's ten worst, which asks whether the WEIGHTS are the "
                        "block-to-block variable")
    p.add_argument("--sweep", action="store_true",
                   help="run the whole arm matrix in one device context")
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", default=OUT)
    a = p.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, os.getcwd())
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import AdaLN, get_device, device_dtype_override
    from tt_bio.openfold3_atom_transformer import remap_of3_adaln
    from tt_bio.taped_ttnn import taped_ttnn
    from tt_bio import taped_ttnn as TT
    from tt_bio.autograd import precise_config, _sum_leading, _wrap, _tape

    # The tape rebinds the name `ttnn` inside every `tt_bio.*` module, and only those. An
    # instrument outside the package holds the REAL module and must address the shim by hand,
    # or its ops run untaped and pybind refuses the wrapper.
    tt = taped_ttnn()

    # ---- the four real weights ---------------------------------------------------------------
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    blocks = sorted({a.block} | (set(PUBLISHED) if a.block_ladder else set()))
    W, PRE, WH = {}, {}, {}
    for b in blocks:
        pre = (f"diffusion_module.diffusion_transformer.blocks.{b}"
               f".attention_pair_bias.layer_norm_a.")
        wb = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
        missing = [k for k in PARAMS if k not in wb]
        if missing:
            print(f"checkpoint has no {missing} under {pre}", file=sys.stderr)
            return 2
        W[b], PRE[b] = wb, pre
        WH[b] = {k: hashlib.sha256(wb[k].double().numpy().tobytes()).hexdigest()[:16]
                 for k in PARAMS}
        print(f"[{time.perf_counter()-t0:.0f}s] block {b} weights: "
              + ", ".join(f"{k} {tuple(wb[k].shape)} {WH[b][k]}" for k in PARAMS), flush=True)
    del sd
    pre, w, wh = PRE[a.block], W[a.block], WH[a.block]

    # ---- the reference class: upstream 0.4.3's own -------------------------------------------
    refpath.install()
    import openfold3
    print(f"REF_TREE resolved: {refpath.assert_resolved()}", flush=True)
    from openfold3.core.model.primitives.normalization import AdaLN as RefAdaLN
    ref_src = os.path.dirname(openfold3.__file__)

    def build_ref(w):
        r = RefAdaLN(c_a=C_A, c_s=C_S, eps=1e-5).double()
        with torch.no_grad():
            r.layer_norm_s.weight.copy_(w["layer_norm_s.weight"].double())
            r.linear_g.weight.copy_(w["linear_g.weight"].double())
            r.linear_g.bias.copy_(w["linear_g.bias"].double())
            r.linear_s.weight.copy_(w["linear_s.weight"].double())
        assert r.linear_s.bias is None and r.layer_norm_a.weight is None, \
            "0.4.3 AdaLN layout changed under this instrument"
        return r

    # ---- AMENDMENT 4: the summand that feeds `_sum_leading` ----------------------------------
    # `autograd.py:1611` is `gamma.add_grad(_sum_leading(ttnn.multiply(g, norm), shape))`.
    # `_sum_leading` passes `precise_config()`, but the PRODUCT that forms its 18,432 summands
    # passes neither a dtype nor a kernel config, so the summands are built at whatever the
    # operands carry and summing them precisely cannot put back bits that were never there.
    # These three rules replace `_taped_layer_norm` from OUTSIDE `tt_bio/`: the shipped one, one
    # that forms the product in fp32 under a precise config, and a deliberately-worse one. A
    # precision change that moves nothing in EITHER direction never reached the kernel.
    # `ttnn.multiply` takes a `dtype` and no `compute_kernel_config` (checked against the
    # binding: the only knobs are dtype, memory_config, activations). So the lever and its
    # control are the same knob in opposite directions, which is the cleanest form of both.
    SUMMAND_DTYPES = []

    def make_ln_rule(kind):
        def rule(shipped, args, kwargs):
            args = list(args) + [None] * (3 - len(args))
            x, gamma, beta = (_wrap(args[0]), _wrap(args[1]), _wrap(args[2]))
            kw = dict(kwargs)
            for nm, slot in (("weight", 1), ("bias", 2)):
                if kw.get(nm) is not None:
                    v = _wrap(kw.pop(nm))
                    gamma, beta = (v, beta) if slot == 1 else (gamma, v)
                else:
                    kw.pop(nm, None)
            kw.pop("l1_headroom", None)
            eps = kw.pop("epsilon", 1e-5)
            ckc = kw.pop("compute_kernel_config", None)
            xv = x.value
            out_v = shipped(xv, weight=(gamma.value if gamma is not None else None),
                            bias=(beta.value if beta is not None else None),
                            epsilon=eps, compute_kernel_config=ckc, **kw)
            bwcfg = precise_config()
            parents = [t for t in (x, gamma, beta) if t is not None]

            def make():
                def bw(g):
                    xv = x.value
                    mean = ttnn.mean(xv, dim=-1, keepdim=True)
                    centered = ttnn.subtract(xv, mean)
                    var = ttnn.mean(ttnn.multiply(centered, centered), dim=-1, keepdim=True,
                                    compute_kernel_config=bwcfg)
                    rstd = ttnn.rsqrt(ttnn.add(var, eps))
                    norm = ttnn.multiply(centered, rstd)
                    if gamma is not None and gamma.requires_grad:
                        if kind == "fp32_summand":
                            pr = ttnn.multiply(g, norm, dtype=ttnn.float32)
                        elif kind == "worse_summand":
                            pr = ttnn.multiply(g, norm, dtype=ttnn.bfloat16)
                        else:
                            pr = ttnn.multiply(g, norm)
                        # The assertion amendment 4 rests on, as a measurement: what dtype the
                        # summands are actually built at on this arm.
                        SUMMAND_DTYPES.append({"kind": kind, "g": str(g.dtype),
                                               "norm": str(norm.dtype),
                                               "product": str(pr.dtype)})
                        gamma.add_grad(_sum_leading(pr, gamma.value.shape))
                    if beta is not None and beta.requires_grad:
                        beta.add_grad(_sum_leading(g, beta.value.shape))
                    if x.requires_grad:
                        dnorm = (ttnn.multiply(g, gamma.value) if gamma is not None else g)
                        dn_mean = ttnn.mean(dnorm, dim=-1, keepdim=True)
                        dn_norm_mean = ttnn.mean(ttnn.multiply(dnorm, norm), dim=-1,
                                                 keepdim=True)
                        dx = ttnn.subtract(ttnn.subtract(dnorm, dn_mean),
                                           ttnn.multiply(norm, dn_norm_mean))
                        x.add_grad(ttnn.multiply(dx, rstd))
                return bw

            return _tape(out_v, parents, make)
        return rule

    shipped_ln = TT._VERBS["layer_norm"]
    LN_RULES = {"shipped": shipped_ln, "fp32_summand": make_ln_rule("fp32_summand"),
                "worse_summand": make_ln_rule("worse_summand")}

    def install_ln(name):
        """`_taped_verb` reads `_VERBS` once and `_Ttnn.__getattr__` caches the wrapper, so the
        registry swap alone is a silent no-op on a verb already called. Both halves, every time."""
        TT._VERBS["layer_norm"] = LN_RULES[name]
        TT._SHIM.__dict__.pop("layer_norm", None)
        ag._TAPED["layer_norm"] = LN_RULES[name]

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def _t(x):
        return ttnn.to_torch(x.value if hasattr(x, "value") else x).double()

    # ------------------------------------------------------------------------------------------
    def run_arm(gate, act_name, samples, s_scale, a_scale, cot_scale, cot_corr, tokens, seed,
                cancel=None, block=None, fp32_weights=False, ln_rule="shipped"):
        block = a.block if block is None else block
        w = W[block]
        N = tokens
        gen = torch.Generator().manual_seed(seed)
        A_in, S_in, COT = [], [], []
        if cancel is not None:
            # The cancellation ladder. `a` and `s` are constant across tokens, so the map from
            # the seeded cotangent to the LayerNorm-gain summand is the SAME linear map at
            # every token; a sign-alternating cotangent then makes the 384 summands cancel
            # exactly and `cancel` is what is left over. K comes out near 20 / cancel, and
            # this is the only knob in the sweep that actually moves it -- the `cot_corr` arms
            # leave it at 20, because the gate path's summand is dominated by the per-token
            # `a_hat` whatever the cotangent does.
            sign = torch.tensor([(-1.0) ** i for i in range(N)]).reshape(1, N, 1)
            for _ in range(samples):
                A_in.append(torch.randn(1, 1, C_A, generator=gen).expand(1, N, C_A).contiguous()
                            * a_scale)
                S_in.append(torch.randn(1, 1, C_S, generator=gen).expand(1, N, C_S).contiguous()
                            * s_scale)
                v = torch.randn(1, 1, C_A, generator=gen)
                COT.append((sign * v + cancel * torch.randn(1, N, C_A, generator=gen))
                           * cot_scale)
        for _ in range(samples if cancel is None else 0):
            A_in.append(torch.randn(1, N, C_A, generator=gen) * a_scale)
            S_in.append(torch.randn(1, N, C_S, generator=gen) * s_scale)
            c = torch.randn(1, N, C_A, generator=gen)
            if cot_corr:
                # One shared row mixed into every token. rho = 1 makes the seeded cotangent
                # identical across tokens, so the gain gradient's per-token terms all point the
                # same way and nothing cancels; rho = 0 leaves them independent, and the sum
                # over N tokens is sqrt(N) times a single term instead of N times it.
                shared = torch.randn(1, 1, C_A, generator=gen).expand(1, N, C_A)
                c = (1.0 - cot_corr ** 2) ** 0.5 * c + cot_corr * shared
            COT.append(c * cot_scale)

        ref = build_ref(w)
        ref_out = []
        for k in range(samples):
            o = ref(A_in[k].double(), S_in[k].double())
            o.backward(COT[k].double())
            ref_out.append(o.detach())
        ref_grad = {n: pm.grad.detach().clone() for n, pm in ref.named_parameters()}

        # The float32 HOST floor: the same class, the same weights, the same inputs, the same
        # cotangent, in torch float32 on the CPU. No device, no tape. It is what single
        # precision alone costs on this comparison, so a device reading at the host floor is
        # not a defect in `tt_bio` however large it is -- and a device reading far above it is.
        refh = build_ref(w).float()
        for k in range(samples):
            refh(A_in[k].float(), S_in[k].float()).backward(COT[k].float())
        host_grad = {n: pm.grad.detach().double() for n, pm in refh.named_parameters()}

        # The gain gradient is a SUM over tokens: g_gamma = sum_i (dL/d ln_out)_i * s_hat_i.
        # K = sum_i ||term_i|| / ||sum_i term_i|| is how much bigger the summands are than
        # their sum, and it is the factor any absolute error in a summand gets amplified by
        # in the RELATIVE reading of the result. Recomputed here from a hand replica, which
        # is checked against the reference class's own output rather than assumed equal.
        F = torch.nn.functional
        gamma = w["layer_norm_s.weight"].double().clone().requires_grad_(True)
        Wg, bg = w["linear_g.weight"].double(), w["linear_g.bias"].double()
        Ws = w["linear_s.weight"].double()
        acc, term_sum, replica = None, 0.0, 0.0
        for k in range(samples):
            shat = F.layer_norm(S_in[k].double(), (C_S,), eps=1e-5)
            ln = shat * gamma
            ahat = F.layer_norm(A_in[k].double(), (C_A,), eps=1e-5)
            o = torch.sigmoid(F.linear(ln, Wg, bg)) * ahat + F.linear(ln, Ws)
            replica = max(replica, _rel(o.detach(), ref_out[k]))
            c = torch.autograd.grad(o, ln, COT[k].double())[0]
            t = (c * shat).reshape(-1, C_S)
            term_sum += float(t.norm(dim=-1).sum())
            acc = t.sum(0) if acc is None else acc + t.sum(0)
        cancellation_K = term_sum / float(acc.norm())

        ag.forget_parameters()
        act = ttnn.float32 if act_name == "fp32" else ttnn.bfloat16
        ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)
        with device_dtype_override(act):
            mod = AdaLN(False, remap_of3_adaln(w), cfg)
        if fp32_weights:
            # `AdaLN.__init__` passes the LITERAL `ttnn.bfloat16` to `torch_to_tt`
            # (tenstorrent.py:9791-9794) instead of routing it through `_dtype`, so all four
            # of its weights stay bf16 even under an fp32 activation override. This arm
            # replaces them in place, outside `tt_bio/`, and answers what that costs.
            f32 = lambda x: ttnn.from_torch(
                (x.t().contiguous() if x.dim() > 1 else x).float(),
                layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.float32)
            mod.s_norm_weight = f32(w["layer_norm_s.weight"])
            mod.s_scale_weight = f32(w["linear_g.weight"])
            mod.s_scale_bias = f32(w["linear_g.bias"])
            mod.s_bias_weight = f32(w["linear_s.weight"])
        dev_of = {"layer_norm_s.weight": mod.s_norm_weight,
                  "linear_g.weight": mod.s_scale_weight,
                  "linear_g.bias": mod.s_scale_bias,
                  "linear_s.weight": mod.s_bias_weight}
        leaves = {n: ag.parameter(t) for n, t in dev_of.items()}

        def gated(A, S):
            if gate == "fused":
                return mod(A, S)
            an = tt.layer_norm(A, epsilon=1e-5,
                               compute_kernel_config=mod.compute_kernel_config)
            s_scale_t, s_bias_t = mod.s_terms(S)
            g_t = tt.sigmoid(s_scale_t)
            an = tt.multiply(an, g_t)
            an = tt.add(an, s_bias_t)
            return tt.to_memory_config(an, memory_config=ttnn.DRAM_MEMORY_CONFIG)

        # A18 gates the gradient, so the discriminator is read off the SAME taped forward the
        # backward is taken from -- a second, untaped forward is a different computation.
        # The accumulation probe rides along: N tapes summing into the same four leaves is
        # exact only if `backward` ADDS across tape contexts, and it watches a leaf EVERY
        # sample touches so a broken accumulation reads as a flat curve instead of nothing.
        fwd, probe = [], []
        install_ln(ln_rule)
        n_dt0 = len(SUMMAND_DTYPES)
        for k in range(samples):
            with device_dtype_override(act), ag.tape():
                out = gated(ag.Tensor(ft(A_in[k])), ag.Tensor(ft(S_in[k])))
                fwd.append(_rel(_t(out).reshape(ref_out[k].shape), ref_out[k]))
                ag.backward([out], [ft(COT[k])])
            gr = leaves["layer_norm_s.weight"].grad
            probe.append(None if gr is None else float(_t(gr).norm()))
        install_ln("shipped")
        seen_dtypes = SUMMAND_DTYPES[n_dt0:n_dt0 + 1]

        rows, zero_rows = [], []
        for n in PARAMS:
            r = ref_grad[n].double()
            gr = leaves[n].grad
            rn = float(r.norm())
            zero_rows.append({"tensor": n, "rel_l2": float(torch.zeros_like(r).sub(r).norm() / rn),
                              "ref_norm": rn})
            if gr is None:
                rows.append({"tensor": n, "rel_l2": None, "absent": True, "ref_norm": rn})
                continue
            d = _t(gr)
            d = d.reshape(d.shape[-r.dim():]) if d.dim() > r.dim() else d
            if tuple(d.shape) != tuple(r.shape) and tuple(d.shape)[::-1] == tuple(r.shape):
                d = d.t().contiguous()
            dn = float(d.norm())
            rl = float((d - r).norm() / (rn + 1e-300))
            rows.append({
                "tensor": n, "shape": list(r.shape), "numel": int(r.numel()), "rel_l2": rl,
                "ref_norm": rn, "device_norm": dn, "norm_ratio": dn / rn if rn else None,
                "cos": float((d * r).sum() / (dn * rn)) if dn and rn else 0.0,
                "ref_sq": rn * rn, "over_bar": bool(rl > PER_TENSOR_BAR),
                "host_fp32_rel_l2": float((host_grad[n] - r).norm() / (rn + 1e-300)),
                "host_fp32_norm_ratio": float(host_grad[n].norm()) / rn if rn else None,
                "host_fp32_cos": (float((host_grad[n] * r).sum()
                                        / (float(host_grad[n].norm()) * rn))
                                  if rn and float(host_grad[n].norm()) else 0.0),
                "published_pct_of_model": PUBLISHED.get(block, {}).get(n)})

        got = [r for r in rows if r.get("rel_l2") is not None]
        den = sum(r["ref_sq"] for r in got)
        mw = ((sum((r["rel_l2"] * r["ref_norm"]) ** 2 for r in got) / den) ** 0.5) if den else None
        hmw = ((sum((r["host_fp32_rel_l2"] * r["ref_norm"]) ** 2 for r in got) / den) ** 0.5) \
            if den else None
        zmw = ((sum((z["rel_l2"] * z["ref_norm"]) ** 2 for z in zero_rows) / den) ** 0.5) \
            if den else None
        worst = max(got, key=lambda r: r["rel_l2"]) if got else None
        arm = {
            "gate": gate, "act": act_name, "block": block, "tokens": N, "samples": samples,
            "s_scale": s_scale, "a_scale": a_scale, "cot_scale": cot_scale,
            "cot_corr": cot_corr, "cancel": cancel, "fp32_weights": fp32_weights,
            "ln_rule": ln_rule, "summand_dtypes": seen_dtypes, "seed": seed,
            "s_scale_is_power_of_two": s_scale in POW2,
            "cot_scale_is_power_of_two": cot_scale in POW2,
            "forward": {"per_sample_rel": fwd, "worst": max(fwd), "bar": PER_TENSOR_BAR},
            "gain_cancellation_K": cancellation_K,
            "replica_vs_reference_class_rel": replica,
            "accumulation_probe": probe,
            "tensors": rows, "mass_weighted_rel": mw,
            "host_fp32_mass_weighted_rel": hmw, "worst": worst,
            "over_bar": sum(1 for r in got if r["over_bar"]),
            "compared": f"{len(got)} of {len(PARAMS)}",
            "zero_model": {"per_tensor": zero_rows, "mass_weighted": zmw},
        }
        print(f"[{time.perf_counter()-t0:.0f}s] b{block} {gate}/{act_name} "
              f"s={s_scale} a={a_scale} " + ("W32 " if fp32_weights else "")
              + (f"ln={ln_rule} " if ln_rule != "shipped" else "") +
              f"cot={cot_scale} rho={cot_corr} cancel={cancel} n={samples}: "
              f"fwd {max(fwd):.4e}  "
              f"mw {mw:.6e} (host fp32 {hmw:.6e}, zero {zmw:.4f})  "
              f"K {cancellation_K:.4g}  "
              f"worst {worst['rel_l2']:.6e} "
              f"{worst['tensor']} r {worst['norm_ratio']:.6f} cos {worst['cos']:.6f}", flush=True)
        for r in got:
            print(f"      {r['tensor']:22s} rel {r['rel_l2']:.6e}  r {r['norm_ratio']:.8f}  "
                  f"cos {r['cos']:.8f}  |g_ref| {r['ref_norm']:.6e}", flush=True)
        return arm

    common = dict(tokens=a.tokens, seed=a.seed)
    arms = []
    if not a.sweep and not a.block_ladder:
        arms = [run_arm(a.gate, a.act, a.samples, a.s_scale, a.a_scale, a.cot_scale,
                        a.cot_corr, cancel=a.cancel, fp32_weights=a.fp32_weights,
                        ln_rule=a.ln_rule, **common)]
    if a.sweep:
        plan = []
        # Deliverable 1 and 3: the base reading, and the gate unfused against the same reference.
        for gate in ("fused", "unfused"):
            plan.append((gate, "fp32", 1, 1.0, 1.0, 1.0, 0.0))
        # Deliverable 2, variable 1: input magnitude. Dirty mantissas (A22), plus a power-of-two
        # `s` arm that is only NEARLY invariant (layer_norm's eps and the sigmoid are not
        # homogeneous in s) and the cotangent power-of-two arm that MUST be bit-identical.
        for k in (0.1, 0.0137, 3.7):
            plan.append(("fused", "fp32", 1, k, 1.0, 1.0, 0.0))
        for k in (0.125, 2.0):
            plan.append(("fused", "fp32", 1, k, 1.0, 1.0, 0.0))
        for k in (0.03125, 32.0):
            plan.append(("fused", "fp32", 1, 1.0, 1.0, k, 0.0))
        for k in (0.0137, 3.7):
            plan.append(("fused", "fp32", 1, 1.0, k, 1.0, 0.0))
        # Deliverable 2, variable 2: the sample axis, accumulating into the same leaves.
        for n in (4, 48):
            plan.append(("fused", "fp32", n, 1.0, 1.0, 1.0, 0.0))
        # The shipped inference dtype, for contrast with the training arm.
        plan.append(("fused", "bf16", 1, 1.0, 1.0, 1.0, 0.0))
        plan.append(("unfused", "bf16", 1, 1.0, 1.0, 1.0, 0.0))
        # The cancellation probe: how correlated the seeded cotangent is across tokens.
        for rho in (0.9, 0.99, 0.999, 1.0):
            plan.append(("fused", "fp32", 1, 1.0, 1.0, 1.0, rho, None))
        # The cancellation ladder, which is the only knob here that moves K.
        for eps in (1.0, 0.1, 0.01, 1e-3, 1e-4, 1e-5):
            plan.append(("fused", "fp32", 1, 1.0, 1.0, 1.0, 0.0, eps))
        plan = [(t + (None,))[:8] for t in plan]
        arms = [run_arm(g, ac, n, ss, asc, cs, cc, cancel=cn, **common)
                for (g, ac, n, ss, asc, cs, cc, cn) in plan]
        # The precision control: the same arms with AdaLN's four weights in fp32 instead of
        # the hardcoded bf16. It prices the one lever this row found, at three conditionings.
        for cn in (None, 1e-3, 1e-5):
            arms.append(run_arm("fused", "fp32", 1, 1.0, 1.0, 1.0, 0.0, cancel=cn,
                                fp32_weights=True, **common))
        # AMENDMENT 4: the summand product, at the rungs where the weight lever died. Both
        # activation dtypes, because the claim is about the ACTIVATION dtype the summands
        # inherit, and the shipped training arm is not the only one that matters.
        for act in ("fp32", "bf16"):
            for cn in (None, 1e-3, 1e-4, 1e-5):
                for lr in ("shipped", "fp32_summand", "worse_summand"):
                    arms.append(run_arm("fused", act, 1, 1.0, 1.0, 1.0, 0.0, cancel=cn,
                                        ln_rule=lr, **common))
    if a.block_ladder:
        # The same inputs and the same cotangent through every block in the campaign's ten
        # worst. If the weights were the block-to-block variable this reads differently per
        # block; if it does not, the variable is what ARRIVES at the block.
        for b in blocks:
            arms.append(run_arm("fused", "fp32", 1, 1.0, 1.0, 1.0, 0.0, block=b, **common))
            arms.append(run_arm("fused", "fp32", 1, 1.0, 1.0, 1.0, 0.0, cancel=1e-3,
                                block=b, **common))

    res = {
        "what": "tenstorrent.AdaLN from one DiT block's real checkpoint weights, taped on "
                "real-shaped (a, s), against a float64 torch autograd reference built from "
                "upstream OpenFold3 0.4.3's own AdaLN class with the same weights, inputs and "
                "cotangent.",
        "weights": {"source": CKPT, "prefix": PRE, "sha256_16": WH},
        "reference": {"class": "openfold3.core.model.primitives.normalization.AdaLN",
                      "version": "0.4.3", "source": ref_src, "dtype": "float64",
                      "engine": "torch autograd"},
        "model_sq_norm": MODEL_SQ_NORM,
        "published_pct_of_model_at_this_block": PUBLISHED.get(a.block),
        "per_tensor_bar": PER_TENSOR_BAR,
        "arms": arms,
        "seconds": time.perf_counter() - t0,
    }
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"adaln_micro{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
