#!/usr/bin/env python3
"""Per-op error INSIDE the template chain, with the reference's z teacher-forced.

The chain reaches 17% relative RMS while every op scores ~1% in isolation. Two
explanations fit: each op injects more error once z has drifted, or the ops inject
~1% each and later ops amplify what earlier ones injected.

This separates them. It walks the ten sub-ops in order and, at each one, feeds the
DEVICE op the REFERENCE's accumulated z rather than the device's own, then scores
that single op's output. Errors cannot accumulate across steps, so whatever shows up
is injection alone.

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/probe_chain_ops.py --ckpt ... --golden ...
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
C = 64

#: (device attribute on PairformerLayer, reference sub-module name)
OPS = [
    ("triangle_multiplication_start", "tri_mul_outgoing"),
    ("triangle_multiplication_end", "tri_mul_incoming"),
    ("triangle_attention_start", "tri_attn_start"),
    ("triangle_attention_end", "tri_attn_end"),
    ("transition_z", "z_transition"),
]


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
    ref = RF3TemplateEmbedder(n_block=2, raw_template_dim=raw_dim, c_z=128, c=C,
                              p_drop=0.0)
    ref.load_state_dict(block_sd, strict=False)
    ref.eval()

    # Replay the reference chain by hand so every intermediate z is available.
    traj = []          # (label, z_before, update_ref)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        feats = template_features(f_in)
        tc = ref.emb_templ(feats)
        v = ref.emb_pair(ref.norm_pair_before_pairformer(z_in)) + tc
        v = v.unsqueeze(0) if v.dim() == 3 else v
        for bi, blk in enumerate(ref.pairformer):
            for _, ref_name in OPS:
                sub = getattr(blk, ref_name)
                upd = sub(v)
                traj.append((f"b{bi}.{ref_name}", v.clone().float(),
                             upd.clone().float()))
                v = v + upd

    mapped = {}
    for key, value in block_sd.items():
        if key.startswith("pairformer."):
            idx, rest = key[len("pairformer."):].split(".", 1)
            for k2, v2 in remap_pairformer_block({rest: value}).items():
                mapped[f"pairformer.{idx}.{k2}"] = v2
        else:
            mapped[key] = value

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    to_tt = lambda x: ttnn.from_torch(
        x.float(), layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    mod = TemplateEmbedder(mapped, cfg)

    rows = []
    for i, (label, z_before, upd_ref) in enumerate(traj):
        bi = int(label[1])
        dev_attr = OPS[i % len(OPS)][0]
        op = getattr(mod.blocks[bi], dev_attr)
        got = torch.Tensor(ttnn.to_torch(op(to_tt(z_before)))).float()
        got = got.reshape(upd_ref.shape)
        diff = (got - upd_ref).abs()
        rows.append({
            "step": i,
            "op": label,
            "pcc": round(pcc(got, upd_ref), 6),
            "rel_rms": round(float(diff.pow(2).mean().sqrt() / upd_ref.std()), 6),
            "z_std": round(float(z_before.std()), 4),
            "update_over_z": round(float(upd_ref.std() / z_before.std()), 4),
        })
        print(json.dumps(rows[-1]), flush=True)

    print(json.dumps({"teacher_forced_per_op": rows}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
