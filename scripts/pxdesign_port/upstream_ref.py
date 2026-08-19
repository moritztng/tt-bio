#!/usr/bin/env python3
"""Run the upstream PXDesign generator on CPU and dump the tensors the port must match.

The captured PD-L1 input (`parity_artifacts/pdl1/ref_design_inputs.pt`) is exactly what
`design_e2e.py` feeds `tt_bio.pxdesign.model.ProtenixDesign`, so both sides start from the
same 17 features and nothing about the comparison depends on a featurizer.

Stages, cheapest first:

  cond     `get_condition_embedding` -> s_inputs (196, 449), s (zeros), z (196, 196, 128).
  denoise  one `diffusion_module` forward at a fixed x_noisy and t_hat, for a few noise
           levels off the real 400-step schedule. This is the sharp test: identical inputs,
           so any difference is the network.
  traj     `sample_diffusion` with a per-step coordinate dump, shared draws.

    ~/protenix_ref_venv/bin/python scripts/pxdesign_port/upstream_ref.py \
        --pxdesign_src ~/pxdesign_src --stage cond --out /tmp/ref_cond.pt
"""
from __future__ import annotations

import argparse
import os
import sys

# protenix's LayerNorm defaults to a CUDA extension it JIT-compiles at import time, which
# needs CUDA_HOME. On CPU it would fall back to the torch implementation anyway, so ask for
# that one up front instead of building a kernel we cannot run.
os.environ.setdefault("LAYERNORM_TYPE", "torch_layernorm")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ART = os.path.join(REPO, "scripts", "pxdesign_port", "parity_artifacts", "pdl1")
CKPT = os.path.expanduser("~/pxdesign_release_data/checkpoint/pxdesign_v0.1.0.pt")

# Stored dtypes in the capture are compact; upstream's own featurizer dtypes are these.
_UPSTREAM_DTYPE = {
    "ref_pos": "float32", "ref_charge": "int64", "ref_element": "int64",
    "ref_atom_name_chars": "int64", "ref_mask": "int64", "ref_space_uid": "int64",
    "atom_to_token_idx": "int64", "restype": "float32", "hotspot": "float32",
    "deletion_mean": "float32", "asym_id": "int64", "residue_index": "int64",
    "entity_id": "int64", "sym_id": "int64", "token_index": "int64",
    "conditional_templ": "int64", "conditional_templ_mask": "bool",
}


def load_inputs():
    import torch
    raw = torch.load(os.path.join(ART, "ref_design_inputs.pt"), weights_only=False)
    return {k: v.to(getattr(torch, _UPSTREAM_DTYPE[k])) for k, v in raw.items()}


def build_model(src, protenix05):
    """Upstream `ProtenixDesign` on CPU with the release checkpoint loaded strictly.

    PXDesign is written against protenix 0.5, whose `AtomAttentionEncoder.forward` takes an
    `input_feature_dict`; the box's installed protenix 2.0 renamed that whole signature, so
    the model half needs the 0.5 modules even though the featurizer capture ran on 2.0.
    `protenix05` is an unpacked protenix 0.5.5 wheel prepended to sys.path -- pure torch,
    no build step, and it leaves the installed 2.0 alone."""
    import torch
    if protenix05:
        assert os.path.isdir(os.path.join(protenix05, "protenix")), protenix05
        sys.path.insert(0, protenix05)
    sys.path.insert(0, HERE)
    import upstream_shim
    upstream_shim.install()          # for the ListValue patch: pxdesign declares empty lists
    sys.path.insert(0, src)
    os.chdir(src)

    from protenix.config import parse_configs
    from pxdesign.configs.configs_base import configs as configs_base
    from pxdesign.configs.configs_data import data_configs
    from pxdesign.configs.configs_infer import inference_configs
    # `eval_configs` comes from the absent pxdbench and nothing in the model reads it.
    cfg = parse_configs(configs={**configs_base, **{"data": data_configs},
                                 **inference_configs},
                        arg_str="", fill_required_with_null=True)
    from pxdesign.model.pxdesign import ProtenixDesign
    model = ProtenixDesign(cfg)
    ck = torch.load(CKPT, map_location="cpu")["model"]
    ck = {k[len("module."):] if k.startswith("module.") else k: v for k, v in ck.items()}
    missing, unexpected = model.load_state_dict(ck, strict=False)
    assert not unexpected, f"checkpoint has keys the model does not: {sorted(unexpected)[:8]}"
    # The two commented-out distogram heads are absent from the checkpoint by design.
    real_missing = [k for k in missing if not k.startswith(("design_distogram_head",
                                                            "design_diffusion_distogram"))]
    assert not real_missing, f"model wants weights the checkpoint lacks: {real_missing[:8]}"
    model.eval()
    return model, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pxdesign_src", default=os.path.expanduser("~/pxdesign_src"))
    ap.add_argument("--protenix05", default=os.path.expanduser("~/protenix05/pkg"),
                    help="unpacked protenix 0.5.5 wheel; PXDesign's pinned model API")
    ap.add_argument("--stage", required=True, choices=("cond", "denoise", "traj"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_step", type=int, default=400)
    ap.add_argument("--steps", default="0,40,120,260,399",
                    help="denoise: which schedule indices to evaluate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_sample", type=int, default=1)
    # The CLI overrides the eta schedule on EVERY real run: `common_run_options` defaults to
    # --eta_type const --eta_min 2.5 --eta_max 2.5 and `parse_sys_args` remaps those onto
    # sample_diffusion.eta_schedule through ALIASES, so configs_base's piecewise_65 never
    # reaches a `pxdesign infer` or `pxdesign pipeline` run. Same three knobs here.
    ap.add_argument("--gamma0", type=float, default=None)
    ap.add_argument("--gamma_min", type=float, default=None)
    ap.add_argument("--eta_type", default=None)
    ap.add_argument("--eta_min", type=float, default=None)
    ap.add_argument("--eta_max", type=float, default=None)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    out_path = os.path.abspath(args.out)
    feats = load_inputs()
    model, cfg = build_model(os.path.abspath(os.path.expanduser(args.pxdesign_src)),
                             os.path.abspath(os.path.expanduser(args.protenix05))
                             if args.protenix05 else None)
    import protenix
    print(f"[ref] protenix from {protenix.__file__}", flush=True)
    if args.gamma0 is not None:
        cfg.sample_diffusion.gamma0 = args.gamma0
    if args.gamma_min is not None:
        cfg.sample_diffusion.gamma_min = args.gamma_min
    if args.eta_type is not None:
        cfg.sample_diffusion.eta_schedule.type = args.eta_type
    if args.eta_min is not None:
        cfg.sample_diffusion.eta_schedule.min = args.eta_min
    if args.eta_max is not None:
        cfg.sample_diffusion.eta_schedule.max = args.eta_max
    print(f"[ref] sample_diffusion: {dict(cfg.sample_diffusion)}", flush=True)
    n_atom = feats["atom_to_token_idx"].shape[0]

    s_inputs, s, z = model.get_condition_embedding(input_feature_dict=feats,
                                                   chunk_size=None)
    rec = {"s_inputs": s_inputs, "s_trunk": s, "z_trunk": z}
    if args.stage == "cond":
        torch.save(rec, out_path)
        print(f"[ref] s_inputs {tuple(s_inputs.shape)} z {tuple(z.shape)} -> {out_path}")
        return

    assert int(cfg.sample_diffusion["N_step"]) == args.n_step, (
        f"config N_step={cfg.sample_diffusion['N_step']} but --n_step={args.n_step}; the "
        "schedule is built here and the eta schedule divides by len(schedule), so they must "
        "agree")
    sched = model.inference_noise_scheduler(N_step=args.n_step, device=s_inputs.device,
                                            dtype=s_inputs.dtype)
    if args.stage == "denoise":
        idx = [int(x) for x in args.steps.split(",")]
        # x_noisy from a fixed seed so the tt-bio side can reproduce it exactly.
        outs = {}
        for k in idx:
            torch.manual_seed(1000 + k)
            x_noisy = sched[k] * torch.randn(1, n_atom, 3)
            t_hat = torch.tensor([float(sched[k])])
            d = model.diffusion_module(x_noisy=x_noisy, t_hat_noise_level=t_hat,
                                       input_feature_dict=feats, s_inputs=s_inputs,
                                       s_trunk=s, z_trunk=z, chunk_size=None,
                                       inplace_safe=False)
            outs[k] = {"sigma": float(sched[k]), "x_noisy": x_noisy, "denoised": d}
            print(f"[ref] denoise step {k}: sigma={float(sched[k]):.4g} "
                  f"|denoised| rms={float(d.pow(2).mean().sqrt()):.4g}", flush=True)
        rec["denoise"] = outs
        rec["sigmas"] = sched
        torch.save(rec, out_path)
        print(f"[ref] -> {out_path}")
        return

    # traj: upstream's OWN `sample_diffusion`, unmodified. Only the denoise net is wrapped,
    # so every constant, the churn, the augmentation and the eta schedule come from upstream.
    frames, calls = {}, {"n": 0}

    def recording_net(**kw):
        d = model.diffusion_module(**kw)
        k = calls["n"]
        frames[k] = {"x_noisy": kw["x_noisy"].clone(), "denoised": d.clone(),
                     "t_hat": kw["t_hat_noise_level"].clone()}
        calls["n"] = k + 1
        if k % 20 == 0 or k == args.n_step - 1:
            print(f"[ref] step {k}/{args.n_step} t_hat={float(kw['t_hat_noise_level'][0]):.4g} "
                  f"rms(denoised)={float(d.pow(2).mean().sqrt()):.4g}", flush=True)
        return d

    torch.manual_seed(args.seed)
    coords = model.sample_diffusion(
        denoise_net=recording_net, input_feature_dict=feats, s_inputs=s_inputs,
        s_trunk=s, z_trunk=z, noise_schedule=sched, N_sample=args.n_sample,
        inplace_safe=False)
    rec["coords"] = coords
    rec["sigmas"] = sched
    rec["frames"] = frames
    rec["n_step"] = args.n_step
    rec["seed"] = args.seed
    torch.save(rec, out_path)
    print(f"[ref] coords {tuple(coords.shape)}, {len(frames)} recorded steps -> {out_path}")


if __name__ == "__main__":
    main()
