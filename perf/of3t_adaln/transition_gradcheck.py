#!/usr/bin/env python3
"""The 23.8917 % no softmax lever touches: the DiT's conditioned transition, alone.

Amendment 6 hands this row the campaign's largest unowned item.
`conditioned_transition.layer_norm.layer_norm_s.weight` is 15.5125 % of the model at
mass-weighted rel 0.1828, 9x over bar, with no softmax above it -- and my sister-AdaLN control
was a valid differential that I should not have called clean.

This runs the transition branch on its own, against upstream 0.4.3's own
`ConditionedTransitionBlock` in float64, on the real `(a, s)` the block hands it -- captured by
a pre-hook on THEIR module while THEIR block runs, so the operands are the model's.

Two input arms, which is the whole point:

  `reference`  the transition is fed their float64 `a`. Everything upstream is exact, so what
               is left is the transition's own arithmetic.
  `ours`       the transition is fed the `a` OUR block actually produces at that point. The
               difference between the two arms is what the attention residual carries in.

Per parameter: `rel_l2`, norm ratio `r` and cosine, with A16's measured zero-model baseline and
the gain sum's cancellation ratio K beside them. Nothing in `tt_bio/` changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_PERF = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PERF not in sys.path:
    sys.path.append(_PERF)
import refpath                                                            # noqa: E402

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
CAP = "/home/ttuser/of3t_diffusion_cap/sub_boundary.pt"
OUT = "perf/of3t_adaln"
PER_TENSOR_BAR = 5.0e-02
MASS_BAR = 2.0e-02


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", default="8")
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--tag", default="")
    p.add_argument("--out-dir", default=OUT)
    a = p.parse_args()
    t0 = time.perf_counter()

    sys.path.insert(0, os.getcwd())
    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio.openfold3_diffusion_transformer import _DiTBlock
    from tt_bio.openfold3_weights import _sub
    from tt_bio.taped_ttnn import taped_ttnn

    # The tape rebinds `ttnn` inside `tt_bio.*` only; an instrument outside the package holds
    # the real module and must address the shim by hand.
    tt = taped_ttnn()

    refpath.install()
    import openfold3
    print(f"REF_TREE resolved: {refpath.assert_resolved()}", flush=True)
    from openfold3.core.model.layers.diffusion_transformer import DiffusionTransformer

    want = [int(x) for x in a.blocks.split(",")]
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    pre = "diffusion_module.diffusion_transformer."
    dsd = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
    del sd
    nb = 1 + max(int(k.split(".")[1]) for k in dsd if k.startswith("blocks."))
    c_a = dsd["blocks.0.attention_pair_bias.mha.linear_q.weight"].shape[1]
    c_s = dsd["blocks.0.attention_pair_bias.layer_norm_a.linear_g.weight"].shape[1]
    c_z = dsd["blocks.0.attention_pair_bias.linear_z.weight"].shape[1]
    no_heads = dsd["blocks.0.attention_pair_bias.linear_z.weight"].shape[0]
    n_trans = dsd["blocks.0.conditioned_transition.swiglu.linear_a.weight"].shape[0] // c_a
    ref_stack = DiffusionTransformer(
        c_a=c_a, c_s=c_s, c_z=c_z, c_hidden=c_a // no_heads, no_heads=no_heads, no_blocks=nb,
        n_transition=n_trans, use_ada_layer_norm=True, n_query=None, n_key=None,
        inf=1e9).double()
    missing, unexpected = ref_stack.load_state_dict(
        {k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
         for k, v in dsd.items()}, strict=False)
    keys = {"missing": len(missing), "unexpected": len(unexpected)}
    if missing or unexpected:
        print("reference key mismatch", keys, file=sys.stderr)
        return 2
    ref_stack.eval()
    print(f"[{time.perf_counter()-t0:.0f}s] reference stack: {keys}", flush=True)

    S = torch.load(CAP, map_location="cpu", weights_only=False)
    dit_args, dit_kw = S["dit_in"]
    del S
    named = {}
    for i, v in enumerate(dit_args):
        named[("a", "s", "z", "mask")[i]] = v
    named.update({k: dit_kw[k] for k in ("a", "s", "z", "mask") if k in dit_kw})

    def one(x):
        while x.dim() > 3 and x.shape[0] == 1:
            x = x[0]
        return x
    A0 = one(named["a"]).double()[:1]
    Sv = one(named["s"]).double()[:1]
    Z = named["z"].double()
    while Z.dim() > 4 and Z.shape[0] == 1:
        Z = Z[0]
    Z = Z[:1]
    N = A0.shape[-2]
    mask = None if named.get("mask") is None else named["mask"].double().reshape(-1, N)[:1]
    print(f"[{time.perf_counter()-t0:.0f}s] operands: a {tuple(A0.shape)} s {tuple(Sv.shape)} "
          f"z {tuple(Z.shape)} N={N}, {0 if mask is None else int(mask.sum())} real tokens",
          flush=True)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    def ft(x):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev,
                               dtype=ttnn.float32)

    def relf(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))

    arms = []
    for b in want:
        blk = ref_stack.blocks[b]
        with torch.no_grad():
            ab = A0
            for i in range(b):
                ab = ref_stack.blocks[i](a=ab, s=Sv, z=Z, mask=mask)

        # THEIR `a` at the transition's own input, captured from THEIR module while THEIR
        # block runs. This is the operand, not a reconstruction of it.
        grab = {}

        def pre(mod, args, kwargs):
            grab["a"] = (kwargs.get("a") if "a" in kwargs else args[0]).detach().clone()

        h = blk.conditioned_transition.register_forward_pre_hook(pre, with_kwargs=True)
        try:
            with torch.no_grad():
                blk(a=ab.clone(), s=Sv, z=Z, mask=mask)
        finally:
            h.remove()
        a_ct_ref = grab["a"]

        # OUR `a` at the same point, from our own block up to the residual.
        bsd = _sub(dsd, f"blocks.{b}")
        with device_dtype_override(ttnn.float32):
            ours = _DiTBlock(bsd, cfg)
        tok = (mask.reshape(-1).float() if mask is not None
               else torch.ones(N, dtype=torch.float32))
        mb = ft(((tok - 1.0) * 1e9).reshape(1, 1, 1, N))
        tmc = ft(tok.reshape(1, N, 1))
        grab_ours = {}
        orig_adaln_t = ours.adaln_t

        class _Spy:
            def __call__(self, A, Sx, **kw):
                grab_ours["a"] = A
                return orig_adaln_t(A, Sx, **kw)
        ours.adaln_t = _Spy()
        with device_dtype_override(ttnn.float32):
            ours(ft(ab), ft(Sv), ft(Z), mb, tmc)
        ours.adaln_t = orig_adaln_t
        a_ct_ours = ttnn.to_torch(
            grab_ours["a"].value if hasattr(grab_ours["a"], "value") else grab_ours["a"]
        ).double().reshape(a_ct_ref.shape)
        residual_rel = relf(a_ct_ours, a_ct_ref)
        print(f"[{time.perf_counter()-t0:.0f}s] block {b}: the transition's own input, ours "
              f"against theirs: {residual_rel:.6e}", flush=True)

        gen = torch.Generator().manual_seed(a.seed)
        cot = torch.randn(a_ct_ref.shape, generator=gen, dtype=torch.float64)

        for src, a_in in (("reference", a_ct_ref), ("ours", a_ct_ours)):
            ct = blk.conditioned_transition
            ct.zero_grad(set_to_none=True)
            taps = {}

            def tap(mod, inp, out):
                out.retain_grad()
                taps["ln"] = {"in": inp[0].detach(), "out": out}

            hh = ct.layer_norm.layer_norm_s.register_forward_hook(tap)
            try:
                out64 = ct(a=a_in.clone(), s=Sv, mask=None)
                out64.backward(cot)
            finally:
                hh.remove()
            ref_grad = {n: pm.grad.detach().clone() for n, pm in ct.named_parameters()
                        if pm.grad is not None}
            gln = taps["ln"]["out"].grad
            shat = torch.nn.functional.layer_norm(
                taps["ln"]["in"], (taps["ln"]["in"].shape[-1],), eps=1e-5)
            terms = (gln * shat).reshape(-1, shat.shape[-1])
            K = float(terms.norm(dim=-1).sum() / (terms.sum(0).norm() + 1e-300))

            ag.forget_parameters()
            with device_dtype_override(ttnn.float32):
                mod = _DiTBlock(bsd, cfg)
            leaves = {}
            for attr, nm in (("s_norm_weight", "layer_norm.layer_norm_s.weight"),
                             ("s_scale_weight", "layer_norm.linear_g.weight"),
                             ("s_scale_bias", "layer_norm.linear_g.bias"),
                             ("s_bias_weight", "layer_norm.linear_s.weight")):
                leaves[nm] = ag.parameter(getattr(mod.adaln_t, attr))
            wmap = {"swiglu.linear_a.weight": "w_la", "swiglu.linear_b.weight": "w_lb",
                    "linear_out.weight": "w_lout", "linear_g.weight": "w_lg",
                    "linear_g.bias": "b_lg"}
            for nm, attr in wmap.items():
                leaves[nm] = ag.parameter(getattr(mod, attr))
            transposed = {nm: True for nm in
                          ("layer_norm.linear_g.weight", "layer_norm.linear_s.weight",
                           "swiglu.linear_a.weight", "swiglu.linear_b.weight",
                           "linear_out.weight", "linear_g.weight")}

            with device_dtype_override(ttnn.float32), ag.tape():
                A = ag.Tensor(ft(a_in))
                Sd = ag.Tensor(ft(Sv))
                a_t = mod.adaln_t(A, Sd)
                b1 = mod._lin(a_t, mod.w_la, activation="silu")
                b2 = mod._lin(a_t, mod.w_lb)
                bb = tt.multiply(b1, b2)
                o = mod._lin(bb, mod.w_lout)
                lg = mod._lin(Sd, mod.w_lg, bias=mod.b_lg)
                o = tt.multiply(o, lg,
                                input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
                fwd = relf(ttnn.to_torch(o.value), out64.detach())
                ag.backward([o], [ft(cot)])

            rows = []
            for nm, leaf in leaves.items():
                r = ref_grad.get(nm)
                g = leaf.grad
                if r is None or g is None:
                    rows.append({"tensor": nm, "rel_l2": None, "absent": True})
                    continue
                r = r.double()
                d = ttnn.to_torch(g).double()
                d = d.reshape(d.shape[-r.dim():]) if d.dim() > r.dim() else d
                if d.dim() == 2 and transposed.get(nm):
                    d = d.t().contiguous()
                if tuple(d.shape) != tuple(r.shape) and tuple(d.shape)[::-1] == tuple(r.shape):
                    d = d.t().contiguous()
                rn, dn = float(r.norm()), float(d.norm())
                rl = float((d - r).norm() / (rn + 1e-300))
                rows.append({"tensor": nm, "rel_l2": rl, "ref_norm": rn, "device_norm": dn,
                             "norm_ratio": dn / rn if rn else None,
                             "cos": float((d * r).sum() / (dn * rn)) if dn and rn else 0.0,
                             "zero_model_rel": float(torch.zeros_like(r).sub(r).norm() / rn),
                             "over_bar": bool(rl > PER_TENSOR_BAR)})
            got = [r for r in rows if r.get("rel_l2") is not None]
            den = sum(r["ref_norm"] ** 2 for r in got)
            mw = ((sum((r["rel_l2"] * r["ref_norm"]) ** 2 for r in got) / den) ** 0.5) \
                if den else None
            gain = next(r for r in got if r["tensor"] == "layer_norm.layer_norm_s.weight")
            arms.append({"block": b, "input": src, "input_rel_vs_reference":
                         (0.0 if src == "reference" else residual_rel),
                         "forward_rel": fwd, "gain_cancellation_K": K,
                         "mass_weighted_rel": mw, "tensors": rows,
                         "over_bar": sum(1 for r in got if r["over_bar"]),
                         "compared": f"{len(got)} of {len(leaves)}",
                         "mass_bar": MASS_BAR, "per_tensor_bar": PER_TENSOR_BAR})
            print(f"[{time.perf_counter()-t0:.0f}s] b{b} input={src:9s} fwd {fwd:.4e}  K {K:.4g}"
                  f"  mw {mw:.6e}  gain rel {gain['rel_l2']:.6e} r {gain['norm_ratio']:.5f} "
                  f"cos {gain['cos']:.6f}  over-bar {arms[-1]['over_bar']}/{len(got)}",
                  flush=True)
            for r in sorted(got, key=lambda x: -x["rel_l2"]):
                print(f"      {r['tensor']:34s} rel {r['rel_l2']:.6e}  r {r['norm_ratio']:9.5f}"
                      f"  cos {r['cos']:9.6f}", flush=True)

    res = {"what": "the DiT conditioned transition alone, ours taped against upstream 0.4.3's "
                   "own ConditionedTransitionBlock in float64, on the real (a, s) the block "
                   "hands it, with the input taken from the reference and from our own block.",
           "reference": {"class": "openfold3.core.model.layers.transition."
                                  "ConditionedTransitionBlock", "version": "0.4.3",
                         "source": os.path.dirname(openfold3.__file__), "dtype": "float64",
                         "keys": keys},
           "capture": CAP, "arms": arms, "seconds": time.perf_counter() - t0}
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"transition_gradcheck{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
