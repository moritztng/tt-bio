#!/usr/bin/env python3
"""The failing tensor's own rule, on the real operands, against float64.

`attention_pair_bias.layer_norm_a.layer_norm_s.weight` is the campaign's worst gradient
disagreement (rel 18.504 at DiT block 8, r 19.24, 8.05 % of the model). Two measurements
already exonerate the AdaLN OP: its sister site `conditioned_transition.layer_norm` is the same
`tt_bio.tenstorrent.AdaLN` class on the same `s` in the same 24 blocks and reads 0.103 at block
8, and the attention tail's data movement gradchecks bit-exact against float64.

This closes the op question directly instead of by inference. It builds ONE AdaLN from the real
checkpoint -- block 8's `attention_pair_bias.layer_norm_a`, the worst site -- hands it the real
conditioned `s` captured at the 0.4.3 boundary and a real-shaped `a`, seeds a random cotangent
at the AdaLN's own output, and compares all four parameter gradients against the same AdaLN
written in torch float64.

If `s_norm.weight` is clean here, the rule is right on the operands it actually fails on, and
the 18.504 is made somewhere between this output and the block's, which is the attention chain.
If it is dirty, the campaign's largest gradient defect is this rule and the search is over.

The cotangent is random rather than the reference's, and that is deliberate: a random seed
tests the LINEAR MAP the backward implements, not one vector through it, and a rule that is
right for a random cotangent is right for every cotangent. The arms vary the `a` operand's
scale, because the one constraint the campaign has is that whatever is wrong scales with
something that varies by block.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.getcwd())

CKPT = os.path.expanduser("~/of3-weights/of3-p2-155k.pt")
CAP = "/home/ttuser/of3t_cond_cap/cond_boundary.pt"
SITE = "diffusion_module.diffusion_transformer.blocks.{b}.attention_pair_bias.layer_norm_a."
SISTER = "diffusion_module.diffusion_transformer.blocks.{b}.conditioned_transition.layer_norm."


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--blocks", default="8,0,12", help="DiT blocks to test")
    p.add_argument("--a-scales", default="1.0,10.0,100.0", dest="a_scales")
    p.add_argument("--act", default="fp32", choices=["fp32", "bf16"])
    p.add_argument("--seed", type=int, default=20260920)
    p.add_argument("--out", default="perf/of3t_conditioning/ADALN_UNIT_GRADCHECK.json")
    a = p.parse_args()
    t0 = time.perf_counter()

    import torch
    import ttnn
    from tt_bio import autograd as ag
    from tt_bio.tenstorrent import AdaLN, get_device, device_dtype_override
    from tt_bio.openfold3_atom_transformer import remap_of3_adaln

    torch.manual_seed(a.seed)
    B = torch.load(CAP, map_location="cpu", weights_only=False)
    s_real = B["si_ref"][0, 0].float()                    # [N_token, c_s], the DiT's own s
    N, C_S = s_real.shape
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
    sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
    print(f"[{time.perf_counter()-t0:.0f}s] s {tuple(s_real.shape)}, checkpoint loaded",
          flush=True)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    act = ttnn.float32 if a.act == "fp32" else ttnn.bfloat16
    ft = lambda x: ttnn.from_torch(x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=act)
    to_t = lambda x: ttnn.to_torch(x.value if hasattr(x, "value") else x).double()

    def rel(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        return float(torch.linalg.vector_norm(x - y)
                     / (torch.linalg.vector_norm(y) + 1e-300))

    def cos(x, y):
        x, y = x.reshape(-1).double(), y.reshape(-1).double()
        nx, ny = x.norm(), y.norm()
        return float((x * y).sum() / (nx * ny)) if nx and ny else 0.0

    def ratio(x, y):
        return float(x.reshape(-1).double().norm() / (y.reshape(-1).double().norm() + 1e-300))

    def ref_adaln(av, sv, g_w, sc_w, sc_b, sb_w):
        """Upstream 0.4.3 AdaLN in float64: weightless a-norm, weight-only s-norm, eps 1e-5,
        sigmoid gate, bias-free shift (normalization.py:88)."""
        an = torch.nn.functional.layer_norm(av, (av.shape[-1],), eps=1e-5)
        sn = torch.nn.functional.layer_norm(sv, (sv.shape[-1],), weight=g_w, eps=1e-5)
        gate = torch.sigmoid(sn @ sc_w.t() + sc_b)
        return gate * an + sn @ sb_w.t()

    results = {}
    for site_name, tmpl in (("attention_pair_bias.layer_norm_a", SITE),
                            ("conditioned_transition.layer_norm", SISTER)):
        for blk in [int(x) for x in a.blocks.split(",")]:
            pre = tmpl.format(b=blk)
            own = {k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}
            if not own:
                results[f"{site_name}.block{blk}"] = {"error": f"no checkpoint keys under {pre}"}
                continue
            mapped = remap_of3_adaln(own)
            with device_dtype_override(act):
                mod = AdaLN(False, mapped, cfg)
            names = ["s_norm.weight", "s_scale.weight", "s_scale.bias", "s_bias.weight"]
            attrs = [mod.s_norm_weight, mod.s_scale_weight, mod.s_scale_bias, mod.s_bias_weight]
            for t in attrs:
                ag.parameter(t)
            g_w, sc_w, sc_b, sb_w = (mapped["s_norm.weight"].double(),
                                     mapped["s_scale.weight"].double(),
                                     mapped["s_scale.bias"].double(),
                                     mapped["s_bias.weight"].double())
            # `a` is the token track, c_a = 768 in the DiT and not c_s; the projection
            # weight's leading dim is the only place in the checkpoint that says so.
            c_a = int(mapped["s_scale.weight"].shape[0])
            for scale in [float(x) for x in a.a_scales.split(",")]:
                at = torch.randn(1, N, c_a) * scale
                gt = torch.randn(1, N, c_a)
                ps = [t.clone().requires_grad_(True) for t in (g_w, sc_w, sc_b, sb_w)]
                yr = ref_adaln(at.double(), s_real.double().unsqueeze(0), *ps)
                yr.backward(gt.double())
                for t in attrs:
                    leaf = ag._PARAMS.get(id(t))
                    if leaf is not None:
                        leaf.grad = None
                with device_dtype_override(act), ag.tape():
                    ad = ag.Tensor(ft(at), requires_grad=True)
                    sd_t = ag.Tensor(ft(s_real.unsqueeze(0)), requires_grad=True)
                    yd = mod(ad, sd_t)
                    fwd = rel(to_t(yd).reshape(yr.shape), yr.detach())
                    ag.backward([yd], [ft(gt)])
                row = {"forward_rel": fwd, "a_scale": scale, "c_a": c_a}
                for nm, t, pr in zip(names, attrs, ps):
                    leaf = ag._PARAMS.get(id(t))
                    gd = getattr(leaf, "grad", None)
                    if gd is None:
                        row[nm] = {"grad": "absent"}
                        continue
                    # `_w_tt` hands the device `w.t().contiguous()`, so a 2-D weight gradient
                    # comes back transposed. Reshaping it instead of transposing it reads
                    # cos ~ 0 with r ~ 1 and rel ~ sqrt(2) on every matrix, at every site, at
                    # every scale -- a constant that looks like a finding and is a harness bug.
                    # `device_gradient.py` transposes back by shape; so does this.
                    gdv = to_t(gd)
                    want = tuple(pr.grad.shape)
                    if gdv.dim() > len(want):
                        gdv = gdv.reshape(gdv.shape[-len(want):])
                    if tuple(gdv.shape) != want and tuple(gdv.shape)[::-1] == want:
                        gdv = gdv.t().contiguous()
                    gdv = gdv.reshape(want)
                    row[nm] = {"rel": rel(gdv, pr.grad), "r": ratio(gdv, pr.grad),
                               "cos": cos(gdv, pr.grad), "ref_norm": float(pr.grad.norm())}
                key = f"{site_name}.block{blk}.a_scale{scale:g}"
                results[key] = row
                worst = max((v["rel"] for v in row.values()
                             if isinstance(v, dict) and "rel" in v), default=None)
                print(f"[{time.perf_counter()-t0:.0f}s] {key:<58} fwd {fwd:.3e} "
                      f"s_norm.weight rel {row['s_norm.weight']['rel']:.3e} "
                      f"r {row['s_norm.weight']['r']:.5f} "
                      f"cos {row['s_norm.weight']['cos']:.6f} | worst {worst:.3e}", flush=True)

    rep = {"what": "our tt_bio.tenstorrent.AdaLN with real checkpoint weights and the real "
                   "captured conditioned `s`, taped and seeded with a RANDOM cotangent at its "
                   "own output, against the same AdaLN in torch float64. A random seed tests "
                   "the linear map the backward implements rather than one vector through it.",
           "reference": "float64 torch, upstream 0.4.3 normalization.py:88 transform",
           "tokens": N, "c_s": C_S, "act": a.act, "seed": a.seed, "bar": 5.0e-02,
           "results": results,
           "over_bar": sorted(k for k, v in results.items() if "error" not in v
                              and max((x["rel"] for x in v.values()
                                       if isinstance(x, dict) and "rel" in x), default=0)
                              > 5.0e-02)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1, sort_keys=True, default=str)
    print("\nover the 5.0e-02 bar:", rep["over_bar"] or "none")
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
