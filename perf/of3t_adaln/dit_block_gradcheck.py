#!/usr/bin/env python3
"""The handoff list, end to end: one real DiT block against upstream 0.4.3's own, in float64.

The sandwich arm (`softmax_sandwich.py`) localised the mechanism in isolation: `ttnn.softmax`
with no `compute_kernel_config` is 2.27e-02 off float64, and the softmax backward
`y*(g - sum(g*y))` amplifies that linearly in the cancellation of `g` along the softmax axis.
This asks the only question that matters after that: does it account for the block's gradient,
and does giving the FORWARD `precise_config()` fix it.

Ours is `tt_bio.openfold3_diffusion_transformer._DiTBlock`, taped, on the real `(a, s, z, mask)`
the diffusion boundary captured. Theirs is
`openfold3.core.model.layers.diffusion_transformer.DiffusionTransformerBlock`, float64, loaded
STRICT from the same checkpoint -- both key sets are reported as numbers, because a reference
that silently drops tensors is how D23/R126 survived forty passes.

The real `a` arriving at block b is produced by running THEIR blocks 0..b-1 in float64, so the
operands are the model's own and not a draw. The cotangent is random with a fixed seed, which
tests the linear map rather than one vector through it.

Nothing in `tt_bio/` changes. The softmax arm is installed into `taped_ttnn._VERBS` from here
and removed again, and the shim's cached wrapper is dropped with it -- without that the patch
is a silent no-op, which is how the first version of the sandwich arm reported two identical
columns.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
REF_PKG = "/home/ttuser/of3t_rebase/of3pkg043"
REF_DEPS = "/home/ttuser/of3t_gradients/pylibs"
CAP = "/home/ttuser/of3t_diffusion_cap/sub_boundary.pt"
OUT = "perf/of3t_adaln"
PER_TENSOR_BAR = 5.0e-02
# The leaf the campaign's worst disagreement sits on, and its sister under the same class.
WATCH = ("attention_pair_bias.layer_norm_a.layer_norm_s.weight",
         "attention_pair_bias.layer_norm_a.linear_g.weight",
         "attention_pair_bias.layer_norm_a.linear_g.bias",
         "attention_pair_bias.layer_norm_a.linear_s.weight",
         "conditioned_transition.layer_norm.layer_norm_s.weight")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", default="8")
    p.add_argument("--k-only", action="store_true", dest="k_only",
                   help="amendment 5: the cancellation ratio K of each AdaLN gain sum on the "
                        "real operands, float64 only, no device arm")
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
    from tt_bio.tenstorrent import get_device, device_dtype_override
    from tt_bio import taped_ttnn as TT
    from tt_bio.taped_ttnn import taped_ttnn
    from tt_bio.openfold3_diffusion_transformer import _DiTBlock
    from tt_bio.openfold3_weights import _sub

    sys.path.insert(0, REF_DEPS)
    sys.path.insert(0, REF_PKG)
    import openfold3
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
    c_hidden = c_a // no_heads
    print(f"[{time.perf_counter()-t0:.0f}s] {nb} blocks, c_a={c_a} c_s={c_s} c_z={c_z} "
          f"heads={no_heads} c_hidden={c_hidden} n_transition={n_trans}", flush=True)

    ref_stack = DiffusionTransformer(
        c_a=c_a, c_s=c_s, c_z=c_z, c_hidden=c_hidden, no_heads=no_heads, no_blocks=nb,
        n_transition=n_trans, use_ada_layer_norm=True, n_query=None, n_key=None,
        inf=1e9).double()
    missing, unexpected = ref_stack.load_state_dict(
        {k: v.double() if torch.is_tensor(v) and v.is_floating_point() else v
         for k, v in dsd.items()}, strict=False)
    keys = {"missing": len(missing), "unexpected": len(unexpected),
            "missing_names": list(missing)[:8], "unexpected_names": list(unexpected)[:8]}
    print(f"[{time.perf_counter()-t0:.0f}s] reference stack loaded: {keys['missing']} missing, "
          f"{keys['unexpected']} unexpected", flush=True)
    if missing or unexpected:
        print("  ", keys, flush=True)
        return 2
    ref_stack.eval()

    S = torch.load(CAP, map_location="cpu", weights_only=False)
    dit_args, dit_kw = S["dit_in"]
    del S
    kw = dict(dit_kw)
    named = {}
    for i, v in enumerate(dit_args):
        named[("a", "s", "z", "mask")[i]] = v
    named.update({k: kw[k] for k in ("a", "s", "z", "mask") if k in kw})
    A0, Sv, Z, M = (named["a"], named["s"], named["z"], named.get("mask"))
    print(f"[{time.perf_counter()-t0:.0f}s] dit_in: a {tuple(A0.shape)} s {tuple(Sv.shape)} "
          f"z {tuple(Z.shape)} mask {None if M is None else tuple(M.shape)}", flush=True)

    # One sample. The 48 noise levels are 48 independent forwards; a rule that is right on
    # one is right on all, and the accumulation across them is already measured exact.
    def one(x):
        while x.dim() > 3 and x.shape[0] == 1:
            x = x[0]
        return x
    A0, Sv = one(A0).double(), one(Sv).double()
    Z = Z.double()
    while Z.dim() > 4 and Z.shape[0] == 1:
        Z = Z[0]
    if A0.dim() == 3 and A0.shape[0] > 1:
        A0, Sv = A0[:1], (Sv[:1] if Sv.shape[0] == A0.shape[0] else Sv)
    if Sv.dim() == 3 and Sv.shape[0] > 1:
        Sv = Sv[:1]
    if Z.dim() == 4 and Z.shape[0] > 1:
        Z = Z[:1]
    N = A0.shape[-2]
    # their `_prep_bias` expands the mask against `a`'s batch dims, so it must be [*, N] with
    # the same number of leading axes as `a` -- the capture carries [1, 1, N].
    mask = None if M is None else M.double().reshape(-1, N)[:1]
    print(f"[{time.perf_counter()-t0:.0f}s] one sample: a {tuple(A0.shape)} s {tuple(Sv.shape)} "
          f"z {tuple(Z.shape)} mask {None if mask is None else tuple(mask.shape)} N={N}",
          flush=True)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    tt = taped_ttnn()
    shipped_rule = TT._VERBS["softmax"]

    def install(rule):
        TT._VERBS["softmax"] = rule
        TT._SHIM.__dict__.pop("softmax", None)

    def host_f64_rule(shipped, args, kwargs):
        """The softmax, forward AND backward, computed on the host in float64. Not a lever --
        it is a BOUND: everything the softmax could contribute is removed. What is left is
        what the rest of the attention path owes."""
        x = TT._wrap(args[0])
        dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
        xv = ttnn.to_torch(x.value).double()
        y64 = torch.softmax(xv, dim=dim)
        y0 = ttnn.from_torch(y64.float(), layout=x.value.layout, device=dev,
                             dtype=x.value.dtype)
        box = [y64]

        def make():
            def bw(g):
                g64 = ttnn.to_torch(g).double()
                yy = box[0]
                d = yy * (g64 - (g64 * yy).sum(dim=dim, keepdim=True))
                x.add_grad(ttnn.from_torch(d.float(), layout=g.layout, device=dev,
                                           dtype=g.dtype))
            return bw

        return TT._tape(y0, [x], make)

    def accurate_forward_rule(shipped, args, kwargs):
        """`tenstorrent._accurate_softmax` in the forward -- max/subtract/exp/sum/divide out
        of individual ops -- with the shipped backward rule unchanged. Unlike the float64
        bound this is a thing that could actually ship, so it is the one worth pricing."""
        from tt_bio.tenstorrent import _accurate_softmax
        x = TT._wrap(args[0])
        dim = kwargs.get("dim", args[1] if len(args) > 1 else -1)
        y0 = _accurate_softmax(x.value, compute_kernel_config=precise_config())
        box = [y0]

        def make():
            def bw(g):
                y = box[0]
                inner = ttnn.sum(ttnn.multiply(g, y), dim=dim, keepdim=True,
                                 compute_kernel_config=precise_config())
                x.add_grad(ttnn.multiply(y, ttnn.subtract(g, inner)))
            return bw

        out = TT._tape(y0, [x], make)
        if out.node is not None:
            out.box = box
        return out

    def precise_forward_rule(shipped, args, kwargs):
        """The shipped tape rule, with `precise_config()` added to the FORWARD softmax call.
        This is the lever the sandwich arm priced; here it runs inside the real block."""
        kwargs = dict(kwargs)
        kwargs.setdefault("compute_kernel_config", precise_config())
        return shipped_rule(shipped, args, kwargs)

    tok = (mask.reshape(-1).float() if mask is not None
           else torch.ones(N, dtype=torch.float32))
    mb = ((tok - 1.0) * 1e9).reshape(1, 1, 1, N)
    tmc = tok.reshape(1, N, 1)

    def ft(x, dt=ttnn.float32):
        return ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)

    def relf(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        return float(torch.linalg.vector_norm(x - y) / (torch.linalg.vector_norm(y) + 1e-300))

    arms = []
    for b in want:
        # THEIR a at block b: run their blocks 0..b-1 in float64 on the captured inputs.
        with torch.no_grad():
            ab = A0
            for i in range(b):
                ab = ref_stack.blocks[i](a=ab, s=Sv, z=Z, mask=mask)
        gen = torch.Generator().manual_seed(a.seed)
        cot = torch.randn(ab.shape, generator=gen, dtype=torch.float64)
        blk = ref_stack.blocks[b]
        blk.zero_grad(set_to_none=True)

        # AMENDMENT 5. The gain gradient is a SUM over tokens,
        # g_gamma = sum_i (dL/d ln_out)_i * s_hat_i, so K = sum_i||term_i|| / ||sum_i term_i||
        # is the amplification any per-summand error gets in the RELATIVE reading of the result.
        # Measured on the reference's own float64 arithmetic at BOTH AdaLN sites, which is what
        # decides whether block 8 is simply higher-K than block 9 or something else.
        taps = {}

        def tap(name):
            def hook(mod, inp, out):
                out.retain_grad()
                taps[name] = {"in": inp[0].detach(), "out": out}
            return hook

        handles = [
            blk.attention_pair_bias.layer_norm_a.layer_norm_s.register_forward_hook(
                tap("attention_pair_bias")),
            blk.conditioned_transition.layer_norm.layer_norm_s.register_forward_hook(
                tap("conditioned_transition")),
        ]
        try:
            out64 = blk(a=ab.clone(), s=Sv, z=Z, mask=mask)
            out64.backward(cot)
        finally:
            for h in handles:
                h.remove()
        ref_grad = {n: pm.grad.detach().clone() for n, pm in blk.named_parameters()
                    if pm.grad is not None}
        kmeas = {}
        for site, t in taps.items():
            gln = t["out"].grad
            shat = torch.nn.functional.layer_norm(t["in"], (t["in"].shape[-1],), eps=1e-5)
            terms = (gln * shat).reshape(-1, shat.shape[-1])
            tot = terms.sum(0)
            kmeas[site] = {"K": float(terms.norm(dim=-1).sum() / (tot.norm() + 1e-300)),
                           "n_summands": int(terms.shape[0]),
                           "sum_norm": float(tot.norm()),
                           "term_norm_sum": float(terms.norm(dim=-1).sum()),
                           "reproduces_gamma_grad": float(
                               (tot - ref_grad[site + (".layer_norm_a" if site ==
                                "attention_pair_bias" else ".layer_norm")
                                + ".layer_norm_s.weight"]).norm()
                               / (tot.norm() + 1e-300))}
        print(f"[{time.perf_counter()-t0:.0f}s] block {b} K: "
              + "  ".join(f"{k} K={v['K']:.4g} (n={v['n_summands']}, "
                          f"self-check {v['reproduces_gamma_grad']:.2e})"
                          for k, v in kmeas.items()), flush=True)
        if a.k_only:
            arms.append({"block": b, "rule": "reference_float64_only", "K": kmeas})
            continue
        print(f"[{time.perf_counter()-t0:.0f}s] block {b}: reference float64 done, "
              f"{len(ref_grad)} parameter gradients", flush=True)

        bsd = _sub(dsd, f"blocks.{b}")
        shape_by_name = {k: tuple(v.shape) for k, v in bsd.items() if torch.is_tensor(v)}
        for rule_name, rule in (("shipped", shipped_rule),
                                ("softmax_precise", precise_forward_rule),
                                ("softmax_accurate", accurate_forward_rule),
                                ("softmax_host_f64", host_f64_rule)):
            install(rule)
            try:
                ag.forget_parameters()
                with device_dtype_override(ttnn.float32):
                    ours = _DiTBlock(bsd, cfg)
                walked = {k: v for k, v in vars(ours).items()
                          if isinstance(v, ttnn.Tensor)}
                # The AdaLNs hold their own weights; reach them by name through the modules.
                for nm, mod in (("adaln_a", ours.adaln_a), ("adaln_t", ours.adaln_t)):
                    for attr in ("s_norm_weight", "s_scale_weight", "s_scale_bias",
                                 "s_bias_weight"):
                        walked[f"{nm}.{attr}"] = getattr(mod, attr)
                leaves = {k: ag.parameter(v) for k, v in walked.items()}
                with device_dtype_override(ttnn.float32), ag.tape():
                    o = ours(ag.Tensor(ft(ab)), ag.Tensor(ft(Sv)), ag.Tensor(ft(Z)),
                             ft(mb), ft(tmc))
                    fwd = relf(ttnn.to_torch(o.value), out64.detach())
                    ag.backward([o], [ft(cot)])
            finally:
                install(shipped_rule)

            # name the device leaves against the checkpoint, by shape-aware mapping
            name_of = {
                "adaln_a.s_norm_weight": "attention_pair_bias.layer_norm_a.layer_norm_s.weight",
                "adaln_a.s_scale_weight": "attention_pair_bias.layer_norm_a.linear_g.weight",
                "adaln_a.s_scale_bias": "attention_pair_bias.layer_norm_a.linear_g.bias",
                "adaln_a.s_bias_weight": "attention_pair_bias.layer_norm_a.linear_s.weight",
                "adaln_t.s_norm_weight": "conditioned_transition.layer_norm.layer_norm_s.weight",
                "adaln_t.s_scale_weight": "conditioned_transition.layer_norm.linear_g.weight",
                "adaln_t.s_scale_bias": "conditioned_transition.layer_norm.linear_g.bias",
                "adaln_t.s_bias_weight": "conditioned_transition.layer_norm.linear_s.weight",
            }
            rows = []
            for dk, nm in name_of.items():
                r = ref_grad.get(nm)
                g = leaves[dk].grad
                if r is None or g is None:
                    rows.append({"tensor": nm, "rel_l2": None, "absent": True})
                    continue
                r = r.double()
                d = ttnn.to_torch(g).double()
                d = d.reshape(d.shape[-r.dim():]) if d.dim() > r.dim() else d
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
            arm = {"block": b, "rule": rule_name, "forward_rel": fwd, "tokens": int(N),
                   "K": kmeas,
                   "mass_weighted_rel": mw, "tensors": rows,
                   "over_bar": sum(1 for r in got if r["over_bar"]),
                   "compared": f"{len(got)} of {len(name_of)}",
                   "reference_keys": keys}
            arms.append(arm)
            w = {r["tensor"]: r for r in got}
            g1 = w.get("attention_pair_bias.layer_norm_a.layer_norm_s.weight")
            g2 = w.get("conditioned_transition.layer_norm.layer_norm_s.weight")
            print(f"[{time.perf_counter()-t0:.0f}s] b{b} {rule_name:16s} fwd {fwd:.4e}  "
                  f"mw {mw:.6e}  apb_gain rel {g1['rel_l2']:.6e} r {g1['norm_ratio']:.5f} "
                  f"cos {g1['cos']:.5f} | ct_gain rel {g2['rel_l2']:.6e} "
                  f"r {g2['norm_ratio']:.5f} cos {g2['cos']:.5f}", flush=True)
            for r in got:
                print(f"      {r['tensor']:62s} rel {r['rel_l2']:.6e}  r {r['norm_ratio']:9.5f}"
                      f"  cos {r['cos']:9.6f}", flush=True)

    res = {"what": "one real DiT block, ours taped against upstream 0.4.3's own in float64, on "
                   "the captured diffusion boundary's real (a, s, z, mask), with and without "
                   "precise_config() on the FORWARD softmax.",
           "reference": {"class": "openfold3.core.model.layers.diffusion_transformer."
                                  "DiffusionTransformerBlock", "version": "0.4.3",
                         "source": os.path.dirname(openfold3.__file__), "dtype": "float64",
                         "keys": keys},
           "capture": CAP, "bar": PER_TENSOR_BAR, "arms": arms,
           "seconds": time.perf_counter() - t0}
    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, f"dit_block_gradcheck{a.tag}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", path, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
