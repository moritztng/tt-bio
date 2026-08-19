#!/usr/bin/env python3
"""On-device parity for RF3's MSA module.

Scores `tt_bio.rf3.msa.MSAModule` against the vendored torch reference at the
module's real operating point, captured by
`capture_trunk_io.py --module recycler.msa_module`. Reports the bf16 ceiling
(torch autocast-bf16 vs torch fp32) on the same input next to the device number,
so a disappointing score can be told apart from a precision floor.

    TT_VISIBLE_DEVICES=2 python scripts/rf3_port/parity_msa.py \\
        --ckpt ... --golden /path/to/msa_io.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

PREFIX = "shadow.recycler.msa_module."


#: Semantic flags on tt-bio's shared blocks. A wrong one is a different computation,
#: not a rounding difference: a missing `fp32_softmax` cost the template embedder 17%
#: relative RMS and took four passes to find, because every probe constructed its own
#: correctly-configured op instead of reading the module's. Echo them in the report so
#: a config drift is visible in every run.
CONFIG_FLAGS = ("fp32_softmax", "scale_pair_bias", "transpose_bias", "ending",
                "biased", "gated_move", "affinity")


def module_config(mod) -> dict:
    """Read the semantic flags off a module's own sub-blocks, recursively."""
    seen = {}

    def walk(obj, path):
        for name in CONFIG_FLAGS:
            if hasattr(obj, name):
                seen[f"{path}.{name}" if path else name] = getattr(obj, name)
        for attr in dir(obj):
            if attr.startswith("_"):
                continue
            try:
                child = getattr(obj, attr)
            except Exception:
                continue
            if isinstance(child, list):
                for i, c in enumerate(child):
                    if hasattr(c, "compute_kernel_config"):
                        walk(c, f"{path}.{attr}[{i}]" if path else f"{attr}[{i}]")
            elif hasattr(child, "compute_kernel_config") and child is not obj:
                walk(child, f"{path}.{attr}" if path else attr)

    walk(mod, "")
    return {k: (bool(v) if isinstance(v, bool) else v) for k, v in sorted(seen.items())}

N_BLOCK = 4


def pcc(a, b) -> float:
    a = a.flatten().double(); b = b.flatten().double()
    a = a - a.mean(); b = b - b.mean()
    d = a.norm() * b.norm()
    return float((a * b).sum() / d) if d else float("nan")


def torch_golden(block_sd, f, z, s_inputs, autocast: bool = True):
    from tt_bio._vendor.rf3.model.layers.pairformer_layers import MSAModule

    # The repo yaml says dim_raw_msa: 34, but the shipped checkpoint and the
    # featurizer both use 35. Read it off the weight so it cannot drift.
    dim_raw_msa = int(block_sd["msa_subsampler.emb_msa.weight"].shape[1])
    mod = MSAModule(
        n_block=N_BLOCK, c_m=64, p_drop_msa=0.0, p_drop_pair=0.0,
        msa_subsample_embedder={"num_sequences": 1024, "dim_raw_msa": dim_raw_msa,
                                "c_s_inputs": 449, "c_msa_embed": 64},
        outer_product={"c_msa_embed": 64, "c_outer_product": 32, "c_out": 128},
        msa_pair_weighted_averaging={"n_heads": 8, "c_weighted_average": 32,
                                     "c_msa_embed": 64, "c_z": 128,
                                     "separate_gate_for_every_channel": True},
        msa_transition={"n": 4, "c": 64},
        triangle_multiplication_outgoing={"d_pair": 128, "d_hidden": 128, "bias": True},
        triangle_multiplication_incoming={"d_pair": 128, "d_hidden": 128, "bias": True},
        triangle_attention_starting={"d_pair": 128, "n_head": 4, "d_hidden": 32, "p_drop": 0.0},
        triangle_attention_ending={"d_pair": 128, "n_head": 4, "d_hidden": 32, "p_drop": 0.0},
        pair_transition={"n": 4, "c": 128},
    )
    missing, unexpected = mod.load_state_dict(block_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"msa_module weights mismatch: {len(missing)} missing, "
                           f"{len(unexpected)} unexpected {(missing or unexpected)[:3]}")
    mod.eval()
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = mod(f, z, s_inputs)
    return out.float()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--golden", required=True)
    args = ap.parse_args()

    import ttnn

    from tt_bio.rf3.msa import MSAModule as TTMSAModule
    from tt_bio.rf3.remap import remap_msa_module
    from tt_bio.tenstorrent import get_device

    sd = torch.load(args.ckpt, map_location="cpu", weights_only=False)["model"]
    block_sd = {k[len(PREFIX):]: v.float() for k, v in sd.items()
                if k.startswith(PREFIX)}

    gold = torch.load(args.golden, weights_only=False)
    f_in, z_in, s_in = gold["in"][0], gold["in"][1], gold["in"][2]
    # the capture keeps non-tensor entries too; only tensors need cloning
    def fresh_f():
        return {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in f_in.items()}

    z_ref = torch_golden(block_sd, fresh_f(), z_in.clone(), s_in.clone())
    z_f32 = torch_golden(block_sd, fresh_f(), z_in.clone(), s_in.clone(),
                         autocast=False)

    dev = get_device()
    cfg = ttnn.init_device_compute_kernel_config(
        dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
        fp32_dest_acc_en=True, packer_l1_acc=True)
    to_tt = lambda x: ttnn.from_torch(
        x.float().unsqueeze(0) if x.dim() in (2, 3) else x.float(),
        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)

    mod = TTMSAModule(N_BLOCK, remap_msa_module(block_sd), cfg)
    z_out = mod(to_tt(f_in["msa"]), to_tt(z_in), to_tt(s_in))
    z_dev = torch.Tensor(ttnn.to_torch(z_out)).float().reshape(z_ref.shape)

    # PCC alone is not a correctness gate: differences near 1.0 are unreadable, and
    # it is blind to a few entries going badly wrong. A missing fp32_softmax read as
    # PCC 0.9993 here while carrying real error, so report relative RMS beside it.
    diff = (z_dev - z_ref).abs()
    ref_diff = (z_ref - z_f32).abs()
    rep = {
        "config": module_config(mod),
        "tokens": int(z_in.shape[-2]),
        "msa_depth": int(f_in["msa"].shape[0]),
        "z_pcc": round(pcc(z_dev, z_ref), 6),
        "z_ceiling_cpu_bf16_vs_fp32": round(pcc(z_ref, z_f32), 6),
        "maxabs": round(float(diff.max()), 6),
        "rel_rms_device": round(float(diff.pow(2).mean().sqrt() / z_ref.std()), 6),
        "rel_rms_reference": round(
            float(ref_diff.pow(2).mean().sqrt() / z_f32.std()), 6),
        "z_ref_std": round(float(z_ref.std()), 4),
    }
    rep["at_ceiling"] = rep["z_pcc"] >= rep["z_ceiling_cpu_bf16_vs_fp32"] - 0.002
    rep["verdict"] = ("PASS" if rep["z_pcc"] > 0.98
                      else "AT_BF16_CEILING" if rep["at_ceiling"] else "GAP")
    print(json.dumps(rep, indent=2))
    return 0 if rep["verdict"] in ("PASS", "AT_BF16_CEILING") else 1


if __name__ == "__main__":
    sys.exit(main())
