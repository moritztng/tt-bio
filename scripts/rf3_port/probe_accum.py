#!/usr/bin/env python3
"""Test whether the template divergence is the residual accumulator's dtype.

torch.autocast casts matmul/linear outputs to bf16 but keeps the residual add and
layer_norm in fp32, so the reference's `z` accumulator is fp32 through all ten
residual updates. tt-bio uploads `z` as bf16 and `ttnn.add_` keeps it there, so
every one of those ten adds rounds.

This runs the template embedder's chain with `z` uploaded as bf16 (current) and as
fp32 (matching the reference) and compares. Everything else is identical.

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/probe_accum.py --ckpt ... --golden ...
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
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        want = ref(dict(f_in), z_in.clone()).float()

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)

    mapped = {}
    for key, value in block_sd.items():
        if key.startswith("pairformer."):
            idx, rest = key[len("pairformer."):].split(".", 1)
            for k2, v2 in remap_pairformer_block({rest: value}).items():
                mapped[f"pairformer.{idx}.{k2}"] = v2
        else:
            mapped[key] = value

    out = {}
    for label, dt in (("bf16 (current)", ttnn.bfloat16), ("fp32", ttnn.float32)):
        try:
            to_tt = lambda x, _dt=dt: ttnn.from_torch(
                x.float().unsqueeze(0) if x.dim() == 3 else x.float(),
                layout=ttnn.TILE_LAYOUT, device=dev, dtype=_dt)
            mod = TemplateEmbedder(mapped, cfg)
            tc = mod.embed_template_feats(to_tt(template_features(f_in)))
            got = torch.Tensor(ttnn.to_torch(
                mod(to_tt(z_in), tc))).float().reshape(want.shape)
            diff = (got - want).abs()
            out[label] = {
                "pcc": round(pcc(got, want), 6),
                "maxabs": round(float(diff.max()), 6),
                "rel_rms": round(float(diff.pow(2).mean().sqrt() / want.std()), 6),
            }
        except Exception as exc:
            out[label] = {"error": f"{type(exc).__name__}: {exc}"[:220]}
        print(json.dumps({label: out[label]}), flush=True)

    print(json.dumps({"tokens": int(z_in.shape[-2]), "activations": out}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
