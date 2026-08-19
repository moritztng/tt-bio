#!/usr/bin/env python3
"""Localise the template embedder's residual to a stage.

The module scores 0.9928 against a bf16 ceiling of 0.999994, where the Pairformer
and MSA module both landed within 1e-3 of theirs. The host-built feature block is
already known bit-exact, so this walks the device path stage by stage against the
reference's own intermediates, captured with hooks.

Stages: emb_templ -> v0 = emb_pair(norm(z)) + template_channels -> pairformer.0 ->
pairformer.1 -> norm_after -> relu -> agg_emb.

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/bisect_template.py \\
        --ckpt ... --golden /path/to/tmpl_io_on.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

PREFIX = "shadow.recycler.template_embedder."


def pcc(a, b) -> float:
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    import ttnn

    from tt_bio._vendor.rf3.model.layers.pairformer_layers import RF3TemplateEmbedder
    from tt_bio.rf3.remap import remap_pairformer_block
    from tt_bio.rf3.template import TemplateEmbedder, template_features
    from tt_bio.tenstorrent import get_device

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    block_sd = {k[len(PREFIX):]: v.float() for k, v in sd.items()
                if k.startswith(PREFIX)}
    gold = torch.load(args.golden, weights_only=False)
    f_in, z_in = gold["in"][0], gold["in"][1]

    raw_dim = int(block_sd["emb_templ.weight"].shape[1])
    ref = RF3TemplateEmbedder(n_block=2, raw_template_dim=raw_dim, c_z=128, c=64,
                              p_drop=0.0)
    ref.load_state_dict(block_sd, strict=False)
    ref.eval()

    taps: dict[str, torch.Tensor] = {}

    def tap(name):
        def hook(_m, _i, o):
            t = o[1] if isinstance(o, tuple) else o
            taps[name] = t.detach().float()
        return hook

    handles = [
        ref.emb_templ.register_forward_hook(tap("emb_templ")),
        ref.pairformer[0].register_forward_hook(tap("pairformer.0")),
        ref.pairformer[1].register_forward_hook(tap("pairformer.1")),
        ref.norm_after_pairformer.register_forward_hook(tap("norm_after")),
        ref.agg_emb.register_forward_hook(tap("agg_emb")),
    ]
    try:
        with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
            ref(dict(f_in), z_in.clone())
    finally:
        for h in handles:
            h.remove()

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    to_tt = lambda x: ttnn.from_torch(
        x.float().unsqueeze(0) if x.dim() == 3 else x.float(),
        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
    back = lambda t, like: torch.Tensor(ttnn.to_torch(t)).float().reshape(like.shape)

    mapped = {}
    for key, value in block_sd.items():
        if key.startswith("pairformer."):
            idx, rest = key[len("pairformer."):].split(".", 1)
            for k2, v2 in remap_pairformer_block({rest: value}).items():
                mapped[f"pairformer.{idx}.{k2}"] = v2
        else:
            mapped[key] = value
    mod = TemplateEmbedder(mapped, cfg)

    out: dict[str, float] = {}

    tc = mod.embed_template_feats(to_tt(template_features(f_in)))
    out["emb_templ"] = round(pcc(back(tc, taps["emb_templ"]), taps["emb_templ"]), 6)

    z_norm = ttnn.layer_norm(to_tt(z_in), weight=mod.pre_norm_weight,
                             bias=mod.pre_norm_bias, epsilon=1e-5,
                             compute_kernel_config=cfg)
    v = ttnn.linear(z_norm, mod.emb_pair_weight, compute_kernel_config=cfg)
    v = ttnn.add_(v, tc)

    for i, block in enumerate(mod.blocks):
        v = block(None, v)[1]
        name = f"pairformer.{i}"
        out[name] = round(pcc(back(v, taps[name]), taps[name]), 6)

    v = ttnn.layer_norm(v, weight=mod.post_norm_weight, bias=mod.post_norm_bias,
                        epsilon=1e-5, compute_kernel_config=cfg)
    out["norm_after"] = round(pcc(back(v, taps["norm_after"]), taps["norm_after"]), 6)

    v = ttnn.relu(v)
    final = ttnn.linear(v, mod.agg_emb_weight, compute_kernel_config=cfg)
    out["agg_emb (final)"] = round(pcc(back(final, taps["agg_emb"]), taps["agg_emb"]), 6)

    print(json.dumps({"tokens": int(z_in.shape[-2]),
                      "active": int(f_in["has_distogram_condition"].sum()),
                      "stages": out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
