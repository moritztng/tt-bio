#!/usr/bin/env python3
"""Alone vs assembled, same block, same crop, SAME PROBE -- the D8 composition discriminator.

R122 measured `tri_att_end / tri_att_start` at 1.03-1.04 with the sub-modules run ALONE
(bisect_grad.json, block 2, 64 tokens, a warmed synthetic probe) and at 3.12-4.35 with the same
modules INSIDE the assembled block (decompose_block0.json, block 0, crop 384 and crop 64). Its
own caveat was that the comparison mixes block index and crop.

Crop is already controlled on disk: at block 0 the assembled ratio is 4.35 at crop 384 and 4.82
at crop 64, so length does not carry it. Block index is sampled at 0 / 23 / 47. What no pair of
artifacts controls is the third difference nobody had written down -- the two arms do not share
a probe. The alone arm draws z from `randn(seed 11) * 0.05` warmed through the preceding blocks
and drives the backward with a norm-matched random cotangent; the assembled arm uses the
reference's own captured activations and the reference's own captured cotangent. An asymmetry
between two axes of the same operation can be a property of the input distribution, so the
comparison that decides D8 has to hold block, crop AND probe fixed and vary only composition.

This runs both arms in one process on one draw:

  ALONE      each TriangleAttention on z, against their float64 module on the same z
  ASSEMBLED  the whole PairFormerBlock on (s, z), against their float64 block on the same (s, z)

Reference is their float64 torch module in both arms (PROTOCOL SS3: never another
approximation). The cotangent is drawn once per output tensor and norm-matched to that output,
so both arms are driven at their own scale.

The ratio is the reading. If end/start stays ~1 alone and jumps assembled on the identical
probe, composition is the mechanism and the probe is bounded out. If it moves alone too, the
3.1x-4.4x on the record is partly a probe artifact and R122's discriminator was under-specified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "perf/of3t_gradients")

OUT_DIR = "perf/of3t_reopen"
BAR, MEDIAN_BAR = 5.0e-02, 2.0e-02
LOGIT = ("linear_q", "linear_k", "linear_z", "layer_norm")
VALUE = ("linear_v", "linear_o", "linear_g")


def rel(a, b):
    import numpy as np
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))


def path_group(k: str) -> str:
    leaf = k.split(".")[-2] if k.endswith((".weight", ".bias")) else k
    for g in LOGIT:
        if leaf.startswith(g):
            return "logit"
    for g in VALUE:
        if leaf.startswith(g):
            return "value"
    return "other"


def stats(d):
    import numpy as np
    v = sorted(d.values())
    if not v:
        return {"n": 0}
    return {"n": len(v), "median": float(np.median(v)), "worst": float(v[-1]),
            "worst_tensor": max(d.items(), key=lambda kv: kv[1])[0],
            "over_bar": int(sum(1 for x in v if x > BAR))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    import numpy as np
    import torch
    import ttnn
    from tt_bio import autograd as ag
    import tt_bio.tenstorrent as T
    from tt_bio.tenstorrent import get_device, device_weights, accurate_softmax_site
    from tt_bio.train.lora import walked_weights
    from tt_bio.openfold3_weights import remap_pairformer_stack, is_openbind
    from openfold3.core.model.layers.triangular_attention import (
        TriangleAttention as TheirTriAtt, TriangleAttentionEndingNode)
    from bijection_device import device_bijection
    from instrument_a_stack import load_ckpt, their_stack

    B, N = a.block, a.tokens
    sd = load_ckpt()
    flat_all = remap_pairformer_stack(sd, prefix="pairformer_stack")
    P = f"pairformer_stack.blocks.{B}."
    L = f"layers.{B}."

    # ---- the one probe both arms see ---------------------------------------------------------
    # bisect_grad.py's construction, reproduced exactly so the ALONE arm below is a control on
    # the artifact already in the branch rather than a new and differently-drawn measurement.
    g = torch.Generator().manual_seed(11)
    s64 = (torch.randn(1, N, 384, generator=g) * 0.05).double()
    z64 = (torch.randn(1, N, N, 128, generator=g) * 0.05).double()
    sm = torch.ones(1, N, dtype=torch.float64)
    pm = torch.ones(1, N, N, dtype=torch.float64)
    warm, dims, _ = their_stack(sd, B, first=0)
    with torch.no_grad():
        for m in warm:
            s64, z64 = m(s64, z64, sm, pm)
    s64, z64 = s64.detach(), z64.detach()
    del warm

    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    ft = lambda x: ttnn.from_torch(x.to(torch.float32), layout=ttnn.TILE_LAYOUT,
                                   device=dev, dtype=ttnn.bfloat16)

    n_heads = flat_all[L + "tri_att_start.linear.weight"].shape[0]
    head_dim = flat_all[L + "tri_att_start.mha.linear_q.weight"].shape[0] // n_heads
    c_z = int(z64.shape[-1])
    no_heads_pair_bias = sd[P + "attn_pair_bias.linear_z.weight"].shape[0]
    c_s = sd[P + "attn_pair_bias.layer_norm_a.weight"].shape[0]
    transpose_bias = not is_openbind(sd)
    acc = accurate_softmax_site("openfold3.trunk")
    SHIPPED_SPB, SHIPPED_TRI_SPB = True, False

    rep = {"instrument": "alone vs assembled on ONE probe -- D8 composition discriminator",
           "block": B, "tokens": N, "bars": {"per_tensor": BAR, "median": MEDIAN_BAR},
           "reference": "their float64 torch module, same probe, both arms",
           "probe": {"seed": 11, "cot_seed": 23, "warmup_blocks": B,
                     "s_norm": float(s64.norm()), "z_norm": float(z64.norm()),
                     "source": "bisect_grad.py's own construction, reproduced"},
           "shipped_config": {"scale_pair_bias": SHIPPED_SPB,
                              "tri_att_scale_pair_bias": SHIPPED_TRI_SPB,
                              "fp32_softmax": True, "transpose_bias": bool(transpose_bias),
                              "accurate_softmax": acc, "source": "openfold3_trunk.py:137"},
           "arms": {}}

    def ours_flat(prefix, strip=""):
        n = len(prefix)
        return {(k[n:][len(strip):] if k[n:].startswith(strip) else k[n:]): v
                for k, v in flat_all.items()
                if k.startswith(prefix) and k.startswith(L)}

    def compare(our_mod, their_mod, g_ref, params):
        dv = {p: ttnn.to_torch(t).to(torch.float32) for p, t in device_weights(our_mod).items()}
        atoms = {n: p.detach().to(torch.float32) for n, p in their_mod.named_parameters()}
        bij = device_bijection(dv, atoms)
        got, miss = {}, []
        for their, pls in bij["placements"].items():
            pick = next((p for p in pls if params.get(p["device_path"]) is not None
                         and params[p["device_path"]].grad is not None), None)
            if pick is None:
                miss.append(their)
                continue
            gd = ttnn.to_torch(params[pick["device_path"]].grad).to(torch.float64)
            band = gd.narrow(pick["axis"], pick["start"], pick["length"])
            if pick["layout"].endswith("transposed"):
                band = band.T
            got[their] = rel(band.contiguous().numpy(), g_ref[their].numpy())
        return got, miss + bij["their_unplaced"]

    # ---- ARM 1: each triangle attention ALONE, on this probe --------------------------------
    alone = {}
    for label, their_ctor, their_pre, our_ctor in (
        ("tri_att_start", lambda: TheirTriAtt(c_in=c_z, c_hidden=head_dim,
                                              no_heads=n_heads, inf=1e9),
         "pair_stack.tri_att_start.",
         lambda: T.TriangleAttention(head_dim, n_heads, False,
                                     ours_flat(L + "tri_att_start.", "mha."), ckc,
                                     scale_pair_bias=SHIPPED_TRI_SPB, fp32_softmax=True)),
        ("tri_att_end", lambda: TriangleAttentionEndingNode(c_in=c_z, c_hidden=head_dim,
                                                            no_heads=n_heads, inf=1e9),
         "pair_stack.tri_att_end.",
         lambda: T.TriangleAttention(head_dim, n_heads, True,
                                     ours_flat(L + "tri_att_end.", "mha."), ckc,
                                     scale_pair_bias=SHIPPED_TRI_SPB, fp32_softmax=True)),
    ):
        them = their_ctor().to(torch.float64)
        them.load_state_dict({k[len(P + their_pre):]: v.to(torch.float64)
                              for k, v in sd.items() if k.startswith(P + their_pre)}, strict=True)
        them.eval()
        for p in them.parameters():
            p.grad = None
        u_ref = them(z64, mask=pm)
        cg = torch.Generator().manual_seed(23)
        cot = torch.randn(*u_ref.shape, generator=cg).double()
        cot = cot / cot.norm() * float(u_ref.detach().norm())
        (u_ref * cot).sum().backward()
        g_ref = {n: p.grad.detach().clone() for n, p in them.named_parameters()}

        mod = our_ctor()
        params = walked_weights(lambda: mod(ft(z64)), None, mod)
        za = ag.Tensor(ft(z64), requires_grad=True)
        with ag.tape():
            u = mod(za)
        fwd = rel(ttnn.to_torch(u.value).to(torch.float64).numpy(), u_ref.detach().numpy())
        ag.backward([u], [ft(cot)])
        got, absent = compare(mod, them, g_ref, params)
        alone[label] = {"forward_rel": fwd, "per_tensor": got, "absent": absent,
                        **stats(got),
                        "logit": stats({k: v for k, v in got.items() if path_group(k) == "logit"}),
                        "value": stats({k: v for k, v in got.items() if path_group(k) == "value"})}
        print(f"ALONE     {label:14s} fwd {fwd:.3e}  median {alone[label]['median']:.4f}  "
              f"worst {alone[label]['worst']:.4f} on {alone[label]['worst_tensor']}", flush=True)
        del them, mod

    # ---- ARM 2: the SAME two modules INSIDE the assembled block, on the SAME probe -----------
    blk = their_stack(sd, 1, first=B)[0][0]
    blk.eval()
    for p in blk.parameters():
        p.grad = None
    s_r, z_r = blk(s64, z64, sm, pm)
    cg = torch.Generator().manual_seed(23)
    cz = torch.randn(*z_r.shape, generator=cg).double()
    cz = cz / cz.norm() * float(z_r.detach().norm())
    cs = torch.randn(*s_r.shape, generator=cg).double()
    cs = cs / cs.norm() * float(s_r.detach().norm())
    ((z_r * cz).sum() + (s_r * cs).sum()).backward()
    g_ref_blk = {n: p.grad.detach().clone() for n, p in blk.named_parameters()
                 if p.grad is not None}

    flat = {f"layers.0." + k[len(L):]: v for k, v in flat_all.items() if k.startswith(L)}
    mod = T.Pairformer(1, head_dim, n_heads, c_s // no_heads_pair_bias, no_heads_pair_bias,
                       True, flat, ckc, scale_pair_bias=SHIPPED_SPB,
                       tri_att_scale_pair_bias=SHIPPED_TRI_SPB, fp32_softmax=True,
                       transpose_bias=transpose_bias, accurate_softmax=acc)
    attn = (1.0 - sm.reshape(1, 1, 1, N)) * -1e9
    params = walked_weights(lambda: mod(ft(s64), ft(z64), ft(pm), ft(attn), ft(attn)), None, mod)
    sa = ag.Tensor(ft(s64), requires_grad=True)
    za = ag.Tensor(ft(z64), requires_grad=True)
    with ag.tape():
        s_out, z_out = mod(sa, za, ft(pm), ft(attn), ft(attn))
    fwd_s = rel(ttnn.to_torch(s_out.value).to(torch.float64).numpy(), s_r.detach().numpy())
    fwd_z = rel(ttnn.to_torch(z_out.value).to(torch.float64).numpy(), z_r.detach().numpy())
    ag.backward([s_out, z_out], [ft(cs), ft(cz)])
    got, absent = compare(mod, blk, g_ref_blk, params)

    def sub(pre):
        n = len(pre)
        return {k[n:]: v for k, v in got.items() if k.startswith(pre)}

    asm = {"forward_s_rel": fwd_s, "forward_z_rel": fwd_z, "n_compared": len(got),
           "absent": absent, "all": stats(got)}
    for label, pre in (("tri_att_start", "pair_stack.tri_att_start."),
                       ("tri_att_end", "pair_stack.tri_att_end.")):
        d = sub(pre)
        asm[label] = {"per_tensor": d, **stats(d),
                      "logit": stats({k: v for k, v in d.items() if path_group(k) == "logit"}),
                      "value": stats({k: v for k, v in d.items() if path_group(k) == "value"})}
        print(f"ASSEMBLED {label:14s} median {asm[label]['median']:.4f}  "
              f"worst {asm[label]['worst']:.4f} on {asm[label]['worst_tensor']}", flush=True)
    print(f"ASSEMBLED block fwd s {fwd_s:.3e} z {fwd_z:.3e}  "
          f"median {asm['all']['median']:.4f} over {asm['all']['over_bar']}/{len(got)}", flush=True)

    rep["arms"] = {"alone": alone, "assembled": asm}
    r_alone = alone["tri_att_end"]["median"] / alone["tri_att_start"]["median"]
    r_asm = asm["tri_att_end"]["median"] / asm["tri_att_start"]["median"]
    rep["ratio_end_over_start"] = {
        "alone_median": r_alone, "assembled_median": r_asm,
        "alone_worst": alone["tri_att_end"]["worst"] / alone["tri_att_start"]["worst"],
        "assembled_worst": asm["tri_att_end"]["worst"] / asm["tri_att_start"]["worst"],
        "composition_factor": r_asm / r_alone}
    print(f"\nend/start  ALONE {r_alone:.3f}   ASSEMBLED {r_asm:.3f}   "
          f"composition {r_asm / r_alone:.2f}x", flush=True)

    out = os.path.join(OUT_DIR, f"compose_triatt_block{B}_n{N}{a.tag}.json")
    os.makedirs(OUT_DIR, exist_ok=True)
    json.dump(rep, open(out, "w"), indent=1)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
