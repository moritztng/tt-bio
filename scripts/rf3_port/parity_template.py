#!/usr/bin/env python3
"""On-device parity for RF3's template embedder.

Scores `tt_bio.rf3.template.TemplateEmbedder` against the vendored torch reference
at the module's real operating point, captured by
`capture_trunk_io.py --module recycler.template_embedder`. Reports the bf16 ceiling
on the same input alongside the device number.

Only the `template` fixture turns this track on (196 non-zero entries in
has_distogram_condition); on the others the module runs on an all-off condition,
which is still worth scoring as the degenerate case but proves nothing about the
template path.
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


def torch_golden(block_sd, f, z, autocast: bool = True):
    from tt_bio._vendor.rf3.model.layers.pairformer_layers import RF3TemplateEmbedder

    raw_dim = int(block_sd["emb_templ.weight"].shape[1])
    mod = RF3TemplateEmbedder(n_block=2, raw_template_dim=raw_dim, c_z=128, c=64,
                              p_drop=0.0)
    missing, unexpected = mod.load_state_dict(block_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"template weights mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected {(missing or unexpected)[:3]}")
    mod.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        return mod(f, z).float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    import ttnn

    from tt_bio.rf3.template import TemplateEmbedder, template_features
    from tt_bio.tenstorrent import get_device

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    block_sd = {k[len(PREFIX):]: v.float() for k, v in sd.items()
                if k.startswith(PREFIX)}

    gold = torch.load(args.golden, weights_only=False)
    f_in, z_in = gold["in"][0], gold["in"][1]

    def fresh_f():
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in f_in.items()}

    ref = torch_golden(block_sd, fresh_f(), z_in.clone())
    ref_f32 = torch_golden(block_sd, fresh_f(), z_in.clone(), autocast=False)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    to_tt = lambda x: ttnn.from_torch(
        x.float().unsqueeze(0) if x.dim() == 3 else x.float(),
        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    # the remap is a pure scope rename plus the shared pairformer leaves
    from tt_bio.rf3.remap import remap_pairformer_block
    mapped = {}
    for key, value in block_sd.items():
        if key.startswith("pairformer."):
            idx, rest = key[len("pairformer."):].split(".", 1)
            for k2, v2 in remap_pairformer_block({rest: value}).items():
                mapped[f"pairformer.{idx}.{k2}"] = v2
        else:
            mapped[key] = value

    mod = TemplateEmbedder(mapped, cfg)
    feats = template_features(f_in)
    tc = mod.embed_template_feats(to_tt(feats))
    out = mod(to_tt(z_in), tc)
    got = torch.Tensor(ttnn.to_torch(out)).float().reshape(ref.shape)

    active = int(f_in["has_distogram_condition"].sum())
    rep = {
        "tokens": int(z_in.shape[-2]),
        "template_entries_active": active,
        "pcc": round(pcc(got, ref), 6),
        "ceiling_cpu_bf16_vs_fp32": round(pcc(ref, ref_f32), 6),
        "ref_std": round(float(ref.std()), 6),
    }
    rep["at_ceiling"] = rep["pcc"] >= rep["ceiling_cpu_bf16_vs_fp32"] - 0.002
    rep["verdict"] = ("PASS" if rep["pcc"] > 0.98
                      else "AT_BF16_CEILING" if rep["at_ceiling"] else "GAP")
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] in ("PASS", "AT_BF16_CEILING") else 1


if __name__ == "__main__":
    sys.exit(main())
