#!/usr/bin/env python3
"""Per-stage output hashes and timings for the fold's host path, so shipped and patched can be
compared across two processes without loading two copies of the model.

Same seed, same checkpoint, same real feats from `cdk2x2_512.yaml`; `z_trunk`/`s_trunk` are
seeded synthetic at the model's shapes. Prints a sha256 per stage output. Identical hashes mean
bit-identical, which is the bar for a host-path rewrite that must not move the CIF.
"""
import argparse
import hashlib
import json
import os
import statistics as st
import sys
import time
from pathlib import Path

# Default to the checkout this file lives in, so the script runs from any worktree on any
# host. A hardcoded /home/moritz/tt-bio is a false negative waiting to happen.
REPO = os.environ.get("B2X_REPO") or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(REPO) / "scripts" / "gpu_vs_tt"))
# B2X_OVERLAY holds a tt_bio package whose boltz2.py is the patched one; it has to win over
# REPO, so it goes in front of both entries above rather than relying on PYTHONPATH (which
# sys.path.insert would jump).
_ov = os.environ.get("B2X_OVERLAY")
if _ov:
    sys.path.insert(0, _ov)
import torch  # noqa: E402

torch.set_grad_enabled(False)
# Not every Boltz-2 submodule's weights are in the checkpoint, so an unseeded process gets a
# different random init and two runs of the SAME code disagree on input_embedder and
# atom_encoder. Seed before the load so the only difference between two runs is the code.
torch.manual_seed(0)


def sha(t):
    x = t.detach().contiguous()
    return hashlib.sha256(x.numpy().tobytes()).hexdigest()[:24]


def timeit(fn, reps, warm=1):
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return st.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import tt_bio.boltz2 as BZ
    import tt_baseline as B
    from tt_bio.main import _resolve_recycling_steps, prepare_features, to_batch
    from tt_bio.worker import _ensure_local_artifacts
    from tt_bio.data.featurizer import Boltz2Featurizer
    from tt_bio.data.mol import load_canonicals
    from tt_bio.data.tokenize import Boltz2Tokenizer

    B.RECYCLING_STEPS = _resolve_recycling_steps(None, "boltz2")
    fix = Path(REPO) / "perf" / "size512" / "fixtures"
    work = Path("/tmp/b2x_hostpath")
    msa_dir, struct_dir = work / f"msa_{a.size}", work / "out"
    msa_dir.mkdir(parents=True, exist_ok=True)
    struct_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(model="boltz2", msa_dir=str(msa_dir), struct_dir=str(struct_dir))
    _ensure_local_artifacts(cfg)
    B.seed_msa_cache(fix / f"cdk2x2_{a.size}.yaml", fix / f"cdk2x2_{a.size}.a3m", msa_dir)
    tk, ft = Boltz2Tokenizer(), Boltz2Featurizer()
    mol_dir = Path(cfg["mol_dir"])
    ccd = load_canonicals(mol_dir)
    feats, _struct = prepare_features(
        fix / f"cdk2x2_{a.size}.yaml", ccd=ccd, mol_dir=mol_dir, msa_dir=msa_dir, tokenizer=tk,
        featurizer=ft, use_msa=True, msa_url="https://api.colabfold.com", msa_strategy="greedy",
        msa_user=None, msa_pass=None, api_key=None, max_msa=8192, msa_db_path=None,
        use_envdb=False, single_sequence=False)
    batch = to_batch(feats, torch.device("cpu"))
    n_tok = int(feats["token_pad_mask"].shape[0])

    _diff = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
             "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
             "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
             "alignment_reverse_diff": True, "synchronize_sigmas": True}
    model = BZ.Boltz2.load_from_checkpoint(
        cfg["conf_ckpt"],
        predict_args={"recycling_steps": B.RECYCLING_STEPS, "sampling_steps": 200,
                      "diffusion_samples": 1, "max_parallel_samples": None},
        diffusion_process_args=_diff,
        pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
        msa_args={"subsample_msa": True, "num_subsampled_msa": 1024,
                  "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4,
                  "msa_dropout": 0.15, "z_dropout": 0.25, "pairwise_head_width": 32,
                  "pairwise_num_heads": 4, "activation_checkpointing": True},
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                       "fk_resampling_interval": 3, "num_gd_steps": 20},
        use_kernels=False, use_tenstorrent=False, trace=False, diffusion_trace=False).eval()

    res = {"module_file": BZ.__file__, "patched": hasattr(BZ, "_bias_stack"),
           "host": os.uname().nodename, "threads": torch.get_num_threads(),
           "n_tokens": n_tok, "n_atoms": int(feats["atom_pad_mask"].shape[0]), "stages": {}}

    relpos = model.rel_pos(batch)
    res["stages"]["rel_pos"] = {"sha": sha(relpos), "shape": list(relpos.shape),
                                "s": round(timeit(lambda: model.rel_pos(batch), a.reps), 5)}
    s_inputs = model.input_embedder(batch)
    res["stages"]["input_embedder"] = {
        "sha": sha(s_inputs), "shape": list(s_inputs.shape),
        "s": round(timeit(lambda: model.input_embedder(batch), a.reps), 5)}

    token_s = model.diffusion_conditioning.atom_encoder.s_to_c_trans[0].normalized_shape[0]
    g = torch.Generator().manual_seed(20260911)
    z_trunk = torch.randn(1, n_tok, n_tok, relpos.shape[-1], generator=g)
    s_tr = torch.randn(1, n_tok, token_s, generator=g)

    pw = model.diffusion_conditioning.pairwise_conditioner(z_trunk, relpos)
    res["stages"]["pairwise_conditioner"] = {
        "sha": sha(pw), "shape": list(pw.shape),
        "s": round(timeit(lambda: model.diffusion_conditioning.pairwise_conditioner(
            z_trunk, relpos), a.reps), 5)}
    del pw

    def dc():
        return model.diffusion_conditioning(
            s_trunk=s_tr, z_trunk=z_trunk, relative_position_encoding=relpos, feats=batch)

    q, c, _tk, aeb, adb, ttb = dc()
    res["stages"]["diffusion_conditioning"] = {
        "s": round(timeit(lambda: dc(), a.reps), 5),
        "sha_q": sha(q), "sha_c": sha(c), "sha_atom_enc_bias": sha(aeb),
        "sha_atom_dec_bias": sha(adb), "sha_token_trans_bias": sha(ttb),
        "shape_token_trans_bias": list(ttb.shape)}

    print(json.dumps(res, indent=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
