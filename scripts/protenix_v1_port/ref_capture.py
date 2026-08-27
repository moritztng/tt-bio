#!/usr/bin/env python3
"""Run upstream Protenix v0.5.0 on CPU over a tt-bio feature dict and dump every tensor
the port has to match.

Reference side of the protenix-v1 parity. Both arms read the SAME feature dict
(dump_feats.py), so what is scored here is the port's arithmetic, not the featurizer;
the featurizer is scored separately.

Runs in its own venv -- upstream's pinned torch==2.3.1 on CPU, biotite==1.0.1 -- and never
touches tt-bio's env:

    ~/protenix05_ref_venv/bin/python scripts/protenix_v1_port/ref_capture.py \
        --feats /tmp/pv1/feats_multimer.pt --out /tmp/pv1/ref_multimer.pt

deepspeed and flash-attn are absent by design: upstream guards every accelerator import with
importlib.util.find_spec, so the forward is pure torch. LAYERNORM_TYPE is pinned to the torch
implementation so no CUDA extension is JIT-built.
"""
from __future__ import annotations

import argparse
import os
import sys

os.environ.setdefault("LAYERNORM_TYPE", "torch_layernorm")

SRC = os.environ.get("PROTENIX_V050_SRC",
                     "/home/moritz/.coworker/protenix-ref-data/src/protenix-0.5.0")
CKPT = os.environ.get("PROTENIX_V1_CKPT",
                      "/home/moritz/.coworker/protenix-ref-data/protenix_base_default_v0.5.0.pt")


def build():
    import torch
    sys.path.insert(0, SRC)
    os.chdir(SRC)
    from configs.configs_base import configs as configs_base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from protenix.config import parse_configs
    configs_base["use_deepspeed_evo_attention"] = False
    cfg = parse_configs(configs={**configs_base, **{"data": data_configs}, **inference_configs},
                        arg_str=[], fill_required_with_null=True)
    from protenix.model.protenix import Protenix
    model = Protenix(cfg).eval()
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["model"]
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    r = model.load_state_dict(sd, strict=True)
    assert not r.missing_keys and not r.unexpected_keys, r
    return model, cfg


def to_upstream(feats):
    """tt-bio feature dict -> upstream input_feature_dict, dtypes as upstream's featurizer emits."""
    import torch
    # `msa` stays an INDEX tensor: upstream one-hots it inside MSAModule (pairformer.py:868),
    # so handing it over as float32 raises "one_hot is only applicable to index tensor".
    f32 = ("ref_pos", "ref_charge", "ref_mask", "restype", "profile", "deletion_mean",
           "has_deletion", "deletion_value", "token_bonds", "distogram_rep_atom_mask",
           "template_distogram", "template_unit_vector", "template_pseudo_beta_mask",
           "template_backbone_frame_mask", "template_aatype")
    i64 = ("msa", "ref_element", "ref_atom_name_chars", "ref_space_uid", "atom_to_token_idx",
           "atom_to_tokatom_idx", "asym_id", "entity_id", "sym_id", "residue_index",
           "token_index", "mol_type")
    out = {}
    for k, v in feats.items():
        if not torch.is_tensor(v):
            out[k] = v
            continue
        out[k] = v.to(torch.float32) if k in f32 else (v.to(torch.long) if k in i64 else v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cycles", type=int, default=None, help="default: the checkpoint's N_cycle")
    ap.add_argument("--steps", type=int, default=0, help="sample_diffusion steps; 0 skips it")
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count() or 8)

    blob = torch.load(args.feats, map_location="cpu", weights_only=False)
    feats = to_upstream(blob["feats"])
    N = feats["residue_index"].shape[-1]
    model, cfg = build()
    n_cycle = args.cycles or cfg.model.N_cycle
    print(f"N_token={N} N_atom={feats['ref_pos'].shape[0]} N_cycle={n_cycle}", flush=True)

    cap = {}
    hooks = []

    def hook(name):
        def fn(_m, inp, out):
            cap.setdefault(name, []).append(
                tuple(t.detach().clone() for t in out) if isinstance(out, tuple)
                else (out.detach().clone() if torch.is_tensor(out) else out))
        return fn

    for name in ("input_embedder", "relative_position_encoding", "msa_module",
                 "pairformer_stack", "template_embedder", "confidence_head",
                 "diffusion_module"):
        hooks.append(getattr(model, name).register_forward_hook(hook(name)))
    # the two atom-level stages inside the diffusion module
    hooks.append(model.diffusion_module.atom_attention_encoder.register_forward_hook(
        hook("diff_atom_encoder")))
    hooks.append(model.diffusion_module.atom_attention_decoder.register_forward_hook(
        hook("diff_atom_decoder")))
    hooks.append(model.diffusion_module.diffusion_transformer.register_forward_hook(
        hook("dit")))
    hooks.append(model.diffusion_module.diffusion_conditioning.register_forward_hook(
        hook("diff_conditioning")))
    hooks.append(model.confidence_head.pairformer_stack.register_forward_hook(
        hook("conf_pairformer")))
    # the trunk-side atom encoder that produces s_inputs
    hooks.append(model.input_embedder.atom_attention_encoder.register_forward_hook(
        hook("trunk_atom_encoder")))

    import time
    # The trunk is 265 s of CPU at N=228. Cache it beside the output so iterating on the
    # diffusion / confidence half does not re-pay it. The cache is keyed on the feature file's
    # content and the cycle count, so it can never be served for a different input.
    import hashlib
    key = hashlib.sha256(open(args.feats, "rb").read()).hexdigest()[:16] + f"-c{n_cycle}"
    tcache = args.out + f".trunk-{key}.pt"
    if os.path.exists(tcache):
        tk = torch.load(tcache, map_location="cpu", weights_only=False)
        s_inputs, s_trunk, z_trunk, cap_trunk = (tk["s_inputs"], tk["s_trunk"], tk["z_trunk"],
                                                 tk["cap"])
        cap.update(cap_trunk)
        print(f"trunk read from cache {tcache}", flush=True)
    else:
        t0 = time.time()
        s_inputs, s_trunk, z_trunk = model.get_pairformer_output(feats, N_cycle=n_cycle)
        print(f"trunk done in {time.time()-t0:.1f}s", flush=True)
        torch.save({"s_inputs": s_inputs, "s_trunk": s_trunk, "z_trunk": z_trunk,
                    "cap": dict(cap)}, tcache)

    out = {"N": N, "n_cycle": n_cycle,
           "s_inputs": s_inputs, "s_trunk": s_trunk, "z_trunk": z_trunk}

    # ---- one denoise at a FIXED x_noisy / t_hat: identical inputs, so any difference is the net
    # DiffusionModule.forward takes x_noisy and returns the DENOISED coordinates (it applies
    # the EDM preconditioning itself), which is exactly the boundary tt_bio's
    # DiffusionModule.denoise returns. f_forward is the inner raw network; do not call that one.
    g = torch.Generator().manual_seed(0)
    n_atom = feats["ref_pos"].shape[0]
    sigma = 40.0
    x_noisy = torch.randn(1, n_atom, 3, generator=g) * sigma
    t_hat = torch.full((1,), sigma)
    cap.pop("diffusion_module", None)
    denoised = model.diffusion_module(
        x_noisy=x_noisy, t_hat_noise_level=t_hat, input_feature_dict=feats,
        s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk, inplace_safe=False, chunk_size=None)
    out.update(x_noisy=x_noisy, t_hat=t_hat, denoised=denoised,
               sigma_data=model.diffusion_module.sigma_data)
    print("denoise done", flush=True)

    # ---- confidence head on the denoised coordinates
    cap.pop("confidence_head", None)
    conf = model.confidence_head(
        input_feature_dict=feats, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
        pair_mask=None, x_pred_coords=denoised,
        use_memory_efficient_kernel=False, use_deepspeed_evo_attention=False, use_lma=False,
        inplace_safe=False, chunk_size=None)
    out["confidence"] = {k: (v.detach().clone() if torch.is_tensor(v) else v)
                         for k, v in (conf.items() if isinstance(conf, dict) else
                                      {"out": conf}.items())}
    print("confidence done", flush=True)

    for h in hooks:
        h.remove()
    out["cap"] = cap
    torch.save(out, args.out)
    print("wrote", args.out, flush=True)
    for k, v in cap.items():
        print(" ", k, len(v), [tuple(t.shape) for t in (v[0] if isinstance(v[0], tuple) else (v[0],))
                               if torch.is_tensor(t)][:3])


if __name__ == "__main__":
    main()
