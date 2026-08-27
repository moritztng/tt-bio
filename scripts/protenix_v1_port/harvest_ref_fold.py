#!/usr/bin/env python3
"""Harvest a protenix-v1 reference FOLD from upstream Protenix v0.5.0 on CPU.

This is the reference side of the `protenix-v1-prot-msa` leg in
`scripts/full_parity_gate.py`. The module-PCC harness beside this file scores stage by
stage against one shared feature dict; this one produces what the release gate needs
instead: a full 200-step, N-sample fold per seed, so the gate can score the device's
best-confidence structure against an external reference the same way the openfold3 and
protenix-v2 legs do.

Three stages, because the two arms cannot share a Python:

  1. `dump_feats.py --a3m` (tt-bio env, host only) writes the feature dict, MSA included.
  2. this script (~/protenix05_ref_venv) runs upstream's trunk, its own
     `sample_diffusion`, and its confidence head, and saves coordinates + confidences.
  3. `--write-cifs` (tt-bio env, host only) turns those coordinates into CIFs through
     tt_bio's own `_write_protenix_structure`, so reference and device structures carry
     identical atom naming and ordering and an RMSD between them is a real RMSD.

Stage 3 is a separate invocation of THIS file so the fixture layout lives in one place.

    ~/protenix05_ref_venv/bin/python scripts/protenix_v1_port/harvest_ref_fold.py \
        --feats /tmp/pv1/feats_prot_msa.pt --seeds 0,1,2,3,4 --steps 200 --samples 5 \
        --out /tmp/pv1/ref_fold

    PYTHONPATH=$WT env/bin/python3 scripts/protenix_v1_port/harvest_ref_fold.py \
        --write-cifs --raw /tmp/pv1/ref_fold --feats /tmp/pv1/feats_prot_msa.pt \
        --fixture docs/implementation-parity-data/ref-fixtures/protenix-v1/prot/\
msa-server_200step_5sample_4cycle_fp32cpu --target-id prot

`--samples` is a measured decision, not a preference: time seed 0 first. Upstream's CPU
fold is 200 network evaluations per sample, and the fixture's directory name states the
sample count, so a fixture whose settings differ from its name is worse than a smaller one.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LAYERNORM_TYPE", "torch_layernorm")

SRC = os.environ.get("PROTENIX_V050_SRC",
                     "/home/moritz/.coworker/protenix-ref-data/src/protenix-0.5.0")
CKPT = os.environ.get("PROTENIX_V1_CKPT",
                      "/home/moritz/.coworker/protenix-ref-data/protenix_base_default_v0.5.0.pt")


def _ref_side(args):
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ref_capture import build, to_upstream  # same loader the module-PCC arm uses

    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("REF_THREADS", os.cpu_count() or 8)))

    blob = torch.load(args.feats, map_location="cpu", weights_only=False)
    feats = to_upstream(blob["feats"])
    n_token = feats["residue_index"].shape[-1]
    n_atom = feats["ref_pos"].shape[0]
    model, cfg = build()
    n_cycle = args.cycles or cfg.model.N_cycle
    print(f"N_token={n_token} N_atom={n_atom} N_cycle={n_cycle} "
          f"steps={args.steps} samples={args.samples}", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # The trunk does not depend on the sampler seed, so it is computed once and reused for
    # every seed. Upstream does the same: runner/inference.py folds all seeds off one
    # get_pairformer_output call.
    t0 = time.time()
    s_inputs, s_trunk, z_trunk = model.get_pairformer_output(feats, N_cycle=n_cycle)
    print(f"trunk done in {time.time() - t0:.1f}s", flush=True)

    from protenix.model.generator import InferenceNoiseScheduler, sample_diffusion

    # `inference_noise_scheduler` is a TOP-LEVEL config block at v0.5.0, not a member of
    # sample_diffusion: s_max 160.0, s_min 4e-4, rho 7, sigma_data 16.0.
    sched = InferenceNoiseScheduler(**dict(cfg.inference_noise_scheduler))

    for seed in args.seeds:
        dst = out / f"seed{seed}"
        if (dst / "raw.pt").exists() and not args.force:
            print(f"seed {seed}: already harvested, skipping", flush=True)
            continue
        dst.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        torch.manual_seed(seed)
        noise = sched(N_step=args.steps, device="cpu", dtype=torch.float32)
        coords = sample_diffusion(
            denoise_net=model.diffusion_module,
            input_feature_dict=feats,
            s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
            noise_schedule=noise, N_sample=args.samples,
            gamma0=cfg.sample_diffusion.gamma0,
            gamma_min=cfg.sample_diffusion.gamma_min,
            noise_scale_lambda=cfg.sample_diffusion.noise_scale_lambda,
            step_scale_eta=cfg.sample_diffusion.step_scale_eta,
            diffusion_chunk_size=None, inplace_safe=False, attn_chunk_size=None)
        t_diff = time.time() - t0
        confs = []
        for k in range(coords.shape[-3]):
            x = coords[..., k, :, :] if coords.dim() == 4 else coords[k]
            c = model.confidence_head(
                input_feature_dict=feats, s_inputs=s_inputs, s_trunk=s_trunk,
                z_trunk=z_trunk, pair_mask=None, x_pred_coords=x.unsqueeze(0),
                use_memory_efficient_kernel=False, use_deepspeed_evo_attention=False,
                use_lma=False, inplace_safe=False, chunk_size=None)
            # v0.5.0's ConfidenceHead returns a TUPLE of logit tensors (plddt, pae, pde),
            # not the dict later releases return. Normalise to a tuple here and let
            # _write_cifs below index it positionally, which is what it already did.
            if isinstance(c, dict):
                c = tuple(c.get("out", (c.get("plddt"), c.get("pae"), c.get("pde"))))
            confs.append(tuple(v.detach().clone() if torch.is_tensor(v) else v for v in c))
        torch.save({"coords": coords, "confidence": confs, "seed": seed,
                    "n_cycle": n_cycle, "steps": args.steps, "samples": args.samples,
                    "n_token": n_token, "n_atom": n_atom,
                    "diffusion_s": t_diff, "total_s": time.time() - t0}, dst / "raw.pt")
        print(f"seed {seed}: {args.samples} samples in {time.time() - t0:.1f}s "
              f"(diffusion {t_diff:.1f}s)", flush=True)
    return 0


def _write_cifs(args):
    """Stage 3: coordinates -> fixture CIFs + results.json, through tt_bio's own writer."""
    import numpy as np
    import torch

    from tt_bio.main import _write_protenix_structure
    from tt_bio.protenix import ConfidenceHead

    # _postprocess touches no instance state -- it only calls the two staticmethods
    # _ptm_iptm / _chain_ptm_iptm -- so a bare instance is enough and stage 3 stays
    # host-only. Scoring the reference's bin LOGITS through the PORT's own postprocess is
    # the same discipline tt_parity.py uses: one function, two inputs.
    post_fn = object.__new__(ConfidenceHead)

    blob = torch.load(args.feats, map_location="cpu", weights_only=False)
    feats = blob["feats"]
    fixture = Path(args.fixture)
    raw_root = Path(args.raw)

    for seed_dir in sorted(raw_root.glob("seed*")):
        raw = torch.load(seed_dir / "raw.pt", map_location="cpu", weights_only=False)
        coords, confs = raw["coords"], raw["confidence"]
        n_sample = len(confs)
        post = []
        for c in confs:
            rc = c.get("out", c) if isinstance(c, dict) else c
            pl, pae_l, pde_l = (rc[0].squeeze(0), rc[1].squeeze(0), rc[2].squeeze(0))
            post.append(post_fn._postprocess(pae_l.float(), pde_l.float(), pl.float(), feats))
        rows = []
        for k in range(n_sample):
            c = post[k]
            rows.append({"plddt": float(c["plddt"]), "ptm": float(c.get("ptm", 0.0)),
                         "iptm": float(c.get("iptm", 0.0))})
        # AF-style ranking, the same expression worker._protenix_emit ranks with.
        def score(r):
            if r["iptm"] > 0.0:
                return 0.8 * r["iptm"] + 0.2 * r["ptm"]
            return r["ptm"] if r["ptm"] > 0.0 else r["plddt"]
        order = sorted(range(n_sample), key=lambda k: score(rows[k]), reverse=True)
        dst = fixture / seed_dir.name
        (dst / "structures").mkdir(parents=True, exist_ok=True)
        for rank, k in enumerate(order):
            x = coords[..., k, :, :] if coords.dim() == 4 else coords[k]
            name = (f"{args.target_id}.cif" if rank == 0
                    else f"{args.target_id}_model_{rank}.cif")
            _write_protenix_structure(x.squeeze(0).float(), feats, None,
                                      dst / "structures" / name, "cif",
                                      b_factors=np.asarray(post[k]["plddt_atom"]) * 100.0)
        best = rows[order[0]]
        (dst / "results.json").write_text(json.dumps({args.target_id: {
            "plddt": round(best["plddt"], 6), "complex_plddt": round(best["plddt"], 6),
            "ptm": round(best["ptm"], 6), "iptm": round(best["iptm"], 6),
            "confidence_score": round(score(best), 6),
            "n_tokens": int(raw["n_token"]), "n_atoms": int(raw["n_atom"]),
            "samples": n_sample, "msa": True}}, indent=2) + "\n")
        (dst / "meta.json").write_text(json.dumps({
            "seed": raw["seed"], "n_cycle": raw["n_cycle"], "steps": raw["steps"],
            "samples": n_sample, "reference_seconds": round(raw["total_s"], 1),
        }, indent=2) + "\n")
        print(f"wrote {dst}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", default=None, help="raw reference dir (ref stage)")
    ap.add_argument("--seeds", default="0", help="comma-separated")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--cycles", type=int, default=None,
                    help="default: the checkpoint's own N_cycle (4 for protenix-v1)")
    ap.add_argument("--force", action="store_true", help="re-harvest an existing seed")
    ap.add_argument("--write-cifs", action="store_true", help="stage 3 (tt-bio env)")
    ap.add_argument("--raw", default=None, help="stage 3: the ref stage's --out")
    ap.add_argument("--fixture", default=None, help="stage 3: fixture directory to fill")
    ap.add_argument("--target-id", default="prot")
    args = ap.parse_args()
    args.seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    if args.write_cifs:
        assert args.raw and args.fixture, "--write-cifs needs --raw and --fixture"
        return _write_cifs(args)
    assert args.out, "the reference stage needs --out"
    return _ref_side(args)


if __name__ == "__main__":
    sys.exit(main())
