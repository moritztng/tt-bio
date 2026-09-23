#!/usr/bin/env python3
"""Lead 1 of the handoff list: the DiT's fp32-cast softmax sandwich, gradchecked in isolation.

`openfold3_diffusion_transformer.py:207-211` is

    sc   = typecast(sc, float32)
    attn = softmax(sc, dim=-1, numeric_stable=True)
    attn = typecast(attn, bfloat16)

and its backward (`taped_ttnn.py:186-206`) is

    inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True)
    dx    = y * (g - inner)

`g - inner` is a NEAR-CANCELLATION: `inner` is the y-weighted mean of `g` along the softmax
axis, so the more uniform `g` is along that axis the more of it cancels, and whatever error is
in `inner` survives at full size into a difference that is small. That `ttnn.sum` is the ONE
reduction on the backward path that takes neither `compute_kernel_config=precise_config()` nor
an fp32 output dtype, while `_sum_leading` (autograd.py:526) takes the first and records the
measurement that motivated it -- cosine 0.379 -> 0.999995 at a 64x64 block.

This arm drives the sandwich alone at the DiT's real shape [1, 16, 384, 384] against a float64
torch reference, sweeps the cancellation directly, and prices the one-line ablation: the same
rule with `precise_config()` on that sum. Nothing in `tt_bio/` changes -- the ablation is
installed into `taped_ttnn._VERBS` from here and removed again.

A22 binds the scale sweep: dirty mantissas, powers of two only as an arithmetic control.
A16's measured zero-model baseline sits beside every reading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

OUT = "perf/of3t_adaln"
N_HEADS, N_TOK = 16, 384
PER_TENSOR_BAR = 5.0e-02
POW2 = {2.0 ** k for k in range(-12, 13)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, default=N_TOK)
    p.add_argument("--heads", type=int, default=N_HEADS)
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", default=OUT)
    a = p.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, os.getcwd())
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.autograd import precise_config
    from tt_bio.tenstorrent import get_device
    from tt_bio import taped_ttnn as TT
    from tt_bio.taped_ttnn import taped_ttnn

    tt = taped_ttnn()
    dev = get_device()
    N, H = a.tokens, a.heads

    # ---- the ablations, installed from here and never in tt_bio/ -----------------------------
    # `_taped_verb` reads `_VERBS` ONCE and `_Ttnn.__getattr__` caches the wrapper in the shim's
    # instance dict, so replacing the registry entry alone is a NO-OP on an already-used verb.
    # The first version of this arm did exactly that and reported two identical columns; the
    # negative control below is what caught it, and it is why the control is in the matrix
    # rather than in a comment.
    shipped_rule = TT._VERBS["softmax"]

    def install(rule):
        TT._VERBS["softmax"] = rule
        TT._SHIM.__dict__.pop("softmax", None)

    def make_rule(kind):
        """`shipped` is the rule as it stands; `precise` gives its reduction
        `precise_config()`; `broken` drops the `inner` term altogether, which is the negative
        control -- a harness that does not move on THAT is not installing anything."""
        def rule(shipped, args, kwargs):
            x = TT._wrap(args[0])
            dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
            ra, rk = TT._raw(args, kwargs)
            y0 = (ttnn.softmax(*ra, **rk) if shipped is ttnn.softmax_in_place
                  else shipped(*ra, **rk))
            box = [y0]

            def make():
                def bw(g):
                    y = box[0]
                    if kind == "broken":
                        x.add_grad(ttnn.multiply(y, g))
                        return
                    kw = {"compute_kernel_config": precise_config()} if kind == "precise" else {}
                    inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True, **kw)
                    x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
                return bw

            out = TT._tape(y0, [x], make)
            if out.node is not None:
                out.box = box
            return out
        return rule

    RULES = {"shipped": shipped_rule, "precise": make_rule("precise"),
             "broken": make_rule("broken")}

    def rel(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))

    def arm(sharp, cancel, rule_name, fwd_cfg):
        """`sharp` scales the scores (softmax sharpness); `cancel` is how much of the seeded
        cotangent varies ALONG the softmax axis. cancel = 0 makes the cotangent constant on
        each row, where the true gradient is exactly zero and the cancellation is total."""
        gen = torch.Generator().manual_seed(a.seed)
        sc = (torch.randn(1, H, N, N, generator=gen) * sharp)
        row = torch.randn(1, H, N, 1, generator=gen)
        g_in = row.expand(1, H, N, N) + cancel * torch.randn(1, H, N, N, generator=gen)
        # bf16 is what the shipped forward hands the sandwich and what the shipped backward
        # hands it back, so both sides are compared at the SAME operands.
        sc = sc.bfloat16().float()
        g_in = g_in.bfloat16().float()

        x64 = sc.double().requires_grad_(True)
        y64 = torch.softmax(x64, dim=-1)
        y64.backward(g_in.double())
        d64 = x64.grad.detach()
        with torch.no_grad():
            inner64 = (g_in.double() * y64.detach()).sum(-1, keepdim=True)
            # How much of the cotangent cancels: the size of `g` against the size of what is
            # left after the weighted mean is removed. This is the amplification any error in
            # `inner` gets in the relative reading of `dx`.
            K = float(torch.linalg.vector_norm(g_in.double())
                      / (torch.linalg.vector_norm(g_in.double() - inner64) + 1e-300))

        install(RULES[rule_name])
        try:
            ag.forget_parameters()
            sc_d = ttnn.from_torch(sc, layout=ttnn.TILE_LAYOUT, device=dev,
                                   dtype=ttnn.bfloat16)
            g_d = ttnn.from_torch(g_in, layout=ttnn.TILE_LAYOUT, device=dev,
                                  dtype=ttnn.bfloat16)
            skw = {"compute_kernel_config": precise_config()} if fwd_cfg == "precise" else {}
            with ag.tape():
                X = ag.Tensor(sc_d, requires_grad=True)
                s32 = tt.typecast(X, ttnn.float32)
                y = tt.softmax(s32, dim=-1, numeric_stable=True, **skw)
                attn = tt.typecast(y, ttnn.bfloat16)
                fwd = rel(ttnn.to_torch(y.value), y64.detach())
                ag.backward([attn], [g_d])
            d_dev = ttnn.to_torch(X.grad).double()
        finally:
            install(shipped_rule)

        rn, dn = float(d64.norm()), float(d_dev.norm())
        r = {"sharp": sharp, "cancel": cancel, "cancellation_K": K,
             "rule": rule_name, "forward_kernel_config": fwd_cfg,
             "sharp_is_power_of_two": sharp in POW2,
             "forward_rel": fwd,
             "rel_l2": float((d_dev - d64).norm() / (rn + 1e-300)),
             "ref_norm": rn, "device_norm": dn, "norm_ratio": dn / rn if rn else None,
             "cos": float((d_dev * d64).sum() / (dn * rn)) if dn and rn else 0.0,
             "zero_model_rel": float(torch.zeros_like(d64).sub(d64).norm() / rn),
             "over_bar": bool(float((d_dev - d64).norm() / (rn + 1e-300)) > PER_TENSOR_BAR)}
        print(f"[{time.perf_counter()-t0:.0f}s] {rule_name:8s} fwd_cfg={fwd_cfg:7s} "
              f"sharp={sharp:<8g} "
              f"cancel={cancel:<8g} K={K:9.4g}  fwd {fwd:.3e}  rel {r['rel_l2']:.6e}  "
              f"r {r['norm_ratio']:9.5f}  cos {r['cos']:9.6f}  zero {r['zero_model_rel']:.4f}",
              flush=True)
        return r

    arms = []
    # The cancellation ladder. Three backward rules and two forward configs on the middle
    # rungs, so the row can say WHICH of the two the number answers to.
    for cancel in (1.0, 0.1, 0.0137, 1e-3, 1e-4):
        arms.append(arm(1.0, cancel, "shipped", "default"))
        arms.append(arm(1.0, cancel, "precise", "default"))
        arms.append(arm(1.0, cancel, "shipped", "precise"))
    # The negative control on the harness itself, on one rung.
    arms.append(arm(1.0, 0.0137, "broken", "default"))
    # A22: softmax sharpness on dirty mantissas, with a power-of-two arithmetic control.
    for sharp in (0.1, 0.0137, 3.7, 11.3):
        arms.append(arm(sharp, 0.0137, "shipped", "default"))
        arms.append(arm(sharp, 0.0137, "shipped", "precise"))
    for sharp in (0.125, 2.0, 8.0):
        arms.append(arm(sharp, 0.0137, "shipped", "default"))

    res = {"what": "the DiT fp32-cast softmax sandwich, alone, at [1, %d, %d, %d], against a "
                   "float64 torch reference, with the shipped backward rule and with the same "
                   "rule's `ttnn.sum` given precise_config()." % (H, N, N),
           "shipped_rule": "taped_ttnn.py:186-206, `ttnn.sum` with no compute_kernel_config",
           "ablations": {
               "precise": "the same backward rule with compute_kernel_config=precise_config() "
                          "on its `ttnn.sum`",
               "broken": "the negative control: the backward drops the `inner` term entirely",
               "forward_precise": "`ttnn.softmax` given compute_kernel_config=precise_config() "
                                  "in the FORWARD, which the shipped DiT does not pass"},
           "note": "all three are installed from this instrument into taped_ttnn._VERBS and the "
                   "shim's cached wrapper is dropped with them; tt_bio/ is unchanged",
           "shape": [1, H, N, N], "bar": PER_TENSOR_BAR, "arms": arms,
           "seconds": time.perf_counter() - t0}
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"softmax_sandwich{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
