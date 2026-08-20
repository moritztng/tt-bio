"""Reproduce and diagnose the Boltz-2 affinity batch-position corruption.

Two byte-identical input YAMLs folded in one process give different affinity
scalars depending on their position in the batch. This script drives the same
``_WorkerState`` path the spawned worker uses, in-process, and records what
differs between target 1 and target N: the global RNG state at each stage
boundary, the structure coordinates, and the affinity features.

    TT_VISIBLE_DEVICES=2 python3 scripts/boltz2_affinity_batch_position_repro.py \
        --aa 256 --n 3 --out /tmp/repro

Add --preload-affinity to load the affinity checkpoint before the first target
instead of lazily during it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def rng_hash() -> str:
    import numpy as np
    import random
    import torch

    h = hashlib.sha256()
    h.update(torch.random.get_rng_state().numpy().tobytes())
    h.update(json.dumps(random.getstate(), default=str).encode())
    st = np.random.get_state()
    h.update(str(st[0]).encode())
    h.update(st[1].tobytes())
    h.update(str(st[2:]).encode())
    return h.hexdigest()[:16]


def tensor_hash(x) -> str:
    import torch

    if not isinstance(x, torch.Tensor):
        return "-"
    return _sha(x.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())


def feats_hash(feats: dict) -> str:
    import torch

    h = hashlib.sha256()
    for k in sorted(feats):
        v = feats[k]
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--aa", type=int, default=256)
    ap.add_argument("--n", type=int, default=3, help="number of identical targets")
    ap.add_argument("--extra-aa", type=int, default=0,
                    help="if >0, append one genuinely different target of this length")
    ap.add_argument("--out", default="/tmp/boltz2_affinity_batch_position")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recycling-steps", type=int, default=3)
    ap.add_argument("--sampling-steps", type=int, default=200)
    ap.add_argument("--diffusion-samples", type=int, default=1)
    ap.add_argument("--sampling-steps-affinity", type=int, default=200)
    ap.add_argument("--diffusion-samples-affinity", type=int, default=5)
    ap.add_argument("--accelerator", default="tenstorrent")
    ap.add_argument("--preload-affinity", action="store_true",
                    help="load the affinity checkpoint before target 1")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "perf" / "nesso1"))

    import torch

    from make_inputs import LADDER_LIGAND, cdk2, yaml_for  # noqa: E402
    from tt_bio.boltz2 import Boltz2  # noqa: E402
    from tt_bio.main import download_all  # noqa: E402
    from tt_bio.worker import _WorkerState  # noqa: E402

    out = Path(args.out)
    (out / "in").mkdir(parents=True, exist_ok=True)
    (out / "structures").mkdir(parents=True, exist_ok=True)
    (out / "msa").mkdir(parents=True, exist_ok=True)

    targets = []
    y = yaml_for(cdk2(args.aa), LADDER_LIGAND)
    for i in range(1, args.n + 1):
        p = out / "in" / f"t{i}.yaml"
        p.write_text(y)
        targets.append(p)
    if args.extra_aa:
        p = out / "in" / f"x{args.extra_aa}.yaml"
        p.write_text(yaml_for(cdk2(args.extra_aa), LADDER_LIGAND))
        targets.append(p)

    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
    cache.mkdir(parents=True, exist_ok=True)
    download_all(cache)

    use_tt = args.accelerator == "tenstorrent"
    _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                  "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0,
                  "sigma_data": 16.0, "P_mean": -1.2, "P_std": 1.5,
                  "coordinate_augmentation": True, "alignment_reverse_diff": True,
                  "synchronize_sigmas": True}
    _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
    _msa = {"subsample_msa": True, "num_subsampled_msa": 1024, "use_paired_feature": True,
            "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15, "z_dropout": 0.25,
            "pairwise_head_width": 32, "pairwise_num_heads": 4,
            "activation_checkpointing": True}
    conf_kwargs = dict(
        predict_args={"recycling_steps": args.recycling_steps,
                      "sampling_steps": args.sampling_steps,
                      "diffusion_samples": args.diffusion_samples,
                      "max_parallel_samples": None},
        diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                       "fk_resampling_interval": 3, "num_gd_steps": 20},
        use_kernels=True, use_tenstorrent=use_tt, trace=False, diffusion_trace=False,
    )
    aff_kwargs = dict(
        predict_args={"recycling_steps": 5, "sampling_steps": args.sampling_steps_affinity,
                      "diffusion_samples": args.diffusion_samples_affinity,
                      "max_parallel_samples": 1},
        diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
        steering_args={"fk_steering": False, "physical_guidance_update": False,
                       "contact_guidance_update": False, "num_particles": 3, "fk_lambda": 4.0,
                       "fk_resampling_interval": 3, "num_gd_steps": 20},
        affinity_mw_correction=True, use_tenstorrent=use_tt, trace=False,
        diffusion_trace=False,
    )
    cfg = {
        "conf_ckpt": str(cache / "boltz2_conf.ckpt"),
        "aff_ckpt": str(cache / "boltz2_aff.ckpt"),
        "conf_kwargs": conf_kwargs, "aff_kwargs": aff_kwargs,
        "mol_dir": str(cache / "mols"), "msa_dir": str(out / "msa"),
        "struct_dir": str(out / "structures"),
        "method": None, "output_format": "cif",
        "write_pae": False, "write_pde": False, "write_embeddings": False,
        "use_msa_server": False, "msa_db_path": None, "use_envdb": False,
        "msa_server_url": None, "msa_pairing_strategy": "greedy",
        "msa_server_username": None, "msa_server_password": None,
        "api_key_value": None, "max_msa_seqs": 8192,
        "fast": False, "single_sequence": True, "seed": args.seed,
        "model": "boltz2",
    }

    state = _WorkerState(args.accelerator)
    state.load_model(cfg)
    state.bind_run("repro", cfg)

    probe: list[dict] = []
    cur: dict = {}

    orig_predict_step = Boltz2.predict_step

    def traced_predict_step(self, batch):
        leg = "affinity" if getattr(self, "affinity_prediction", False) else "structure"
        cur[f"rng_before_{leg}_fwd"] = rng_hash()
        cur[f"feats_{leg}"] = feats_hash(batch)
        outp = orig_predict_step(self, batch)
        cur[f"rng_after_{leg}_fwd"] = rng_hash()
        if leg == "structure" and "coords" in outp:
            cur["coords"] = tensor_hash(outp["coords"])
        return outp

    Boltz2.predict_step = traced_predict_step

    if args.preload_affinity:
        state.aff_model = (
            Boltz2.load_from_checkpoint(cfg["aff_ckpt"], **cfg["aff_kwargs"])
            .eval()
            .to(state.torch_device)
        )
        print("[repro] affinity checkpoint preloaded", flush=True)

    import time

    for idx, path in enumerate(targets, 1):
        cur = {"pos": idx, "target": path.name}
        t0 = time.time()
        cur["aff_model_loaded_before"] = state.aff_model is not None
        metrics, best, feats = state.predict_one(path, cfg)
        cur["rng_after_structure_job"] = rng_hash()
        aff = state.predict_affinity(path, best, cfg)
        cur["rng_after_affinity_job"] = rng_hash()
        cur.update({k: v for k, v in aff.items()})
        cur["wall_s"] = round(time.time() - t0, 1)
        probe.append(dict(cur))
        print("[repro] " + json.dumps(cur), flush=True)

    print()
    keys = ["pos", "target", "aff_model_loaded_before", "coords",
            "rng_before_structure_fwd", "rng_before_affinity_fwd",
            "feats_structure", "feats_affinity",
            "affinity_pred_value", "affinity_probability_binary", "wall_s"]
    w = {k: max(len(k), *(len(str(r.get(k, "-"))) for r in probe)) for k in keys}
    print(" | ".join(k.ljust(w[k]) for k in keys))
    for r in probe:
        print(" | ".join(str(r.get(k, "-")).ljust(w[k]) for k in keys))

    vals = [r.get("affinity_pred_value") for r in probe[: args.n]]
    same = len(set(vals)) == 1
    print(f"\nIDENTICAL-TARGET AFFINITY SCALARS AGREE: {same}  {vals}")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(probe, indent=2))
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
