#!/usr/bin/env python3
"""One Boltz-2 fold arm, on whichever ttnn stack the interpreter that runs it was built with.

The A/B this task owes cannot be a single process: the two arms are two different `ttnn` shared
objects and only one can be loaded per interpreter. So the pairing moves up one level -- this
script is one ARM of one ROUND, the driver (`drive_ab.py`) alternates rounds, and every fold is
stamped with its round index so the pairing is auditable after the fact rather than assumed.

Everything else follows the published 512 aa protocol: `perf/size512/fixtures/cdk2x2_512.yaml`
with its fixed 35-row a3m, 3 recycles, 200 sampling steps, 1 diffusion sample, seed 0, templates
off, timed around `_WorkerState.predict_one`, first fold of the process discarded as cold.

Writes one JSON line per fold to --out, flushed immediately, so a killed run keeps what it measured.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
FIX = REPO / "perf" / "size512" / "fixtures"

RECYCLING_STEPS = 3
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
SEED = 0


def _seed_msa(target: Path, a3m_text: str, msa_dir: Path) -> None:
    from tt_bio.main import _read_bio_chains
    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    assert a3m_text.split("\n")[1] == seq, "a3m query row does not match the target sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / f"{h}.a3m").write_text(a3m_text)


def build_cfg(msa_dir: Path, struct_dir: Path, model: str = "boltz2") -> dict:
    return dict(
        model=model, fast=False, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
        conf_kwargs=dict(
            predict_args={"recycling_steps": RECYCLING_STEPS,
                          "sampling_steps": SAMPLING_STEPS,
                          "diffusion_samples": DIFFUSION_SAMPLES,
                          "max_parallel_samples": 5},
            diffusion_process_args={
                "step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001,
                "sigma_max": 160.0, "sigma_data": 16.0, "P_mean": -1.2,
                "P_std": 1.5, "coordinate_augmentation": True,
                "alignment_reverse_diff": True, "synchronize_sigmas": True},
            pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
            msa_args={"subsample_msa": False, "num_subsampled_msa": 1024,
                      "use_paired_feature": True, "msa_s": 64, "msa_blocks": 4,
                      "msa_dropout": 0.15, "z_dropout": 0.25,
                      "pairwise_head_width": 32, "pairwise_num_heads": 4,
                      "activation_checkpointing": True},
            steering_args={"fk_steering": False, "physical_guidance_update": False,
                           "contact_guidance_update": True, "num_particles": 3,
                           "fk_lambda": 4.0, "fk_resampling_interval": 3,
                           "num_gd_steps": 20},
            use_kernels=True, use_tenstorrent=True, trace=False,
            diffusion_trace=False,
        ),
    )


def main() -> int:
    global SAMPLING_STEPS, RECYCLING_STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True, help="JSONL, appended")
    ap.add_argument("--arm", required=True, help="label, e.g. old / new")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--folds", type=int, default=2, help="TIMED folds; the cold one is extra")
    ap.add_argument("--fixture", default="cdk2x2_512")
    ap.add_argument("--steps", type=int, default=SAMPLING_STEPS)
    ap.add_argument("--recycles", type=int, default=RECYCLING_STEPS)
    ap.add_argument("--cifdir", type=Path, default=None, help="keep every fold's CIF here")
    ap.add_argument("--model", default="boltz2",
                    help="blast radius: the same stack A/B through another model")
    args = ap.parse_args()
    SAMPLING_STEPS, RECYCLING_STEPS = args.steps, args.recycles

    import torch
    torch.set_grad_enabled(False)
    import ttnn
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    _E.set_progress(lambda *a, **k: None)
    import tt_bio as _TB
    assert Path(_TB.__file__).resolve().is_relative_to(REPO), (
        f"imported tt_bio from {_TB.__file__}, not {REPO}")

    import importlib.metadata as _md
    ttnn_version = _md.version("ttnn")

    dev = get_device()
    try:
        g = dev.compute_with_storage_grid_size()
        grid = [g.x, g.y]
    except Exception:
        grid = None

    work = Path(tempfile.mkdtemp(prefix=f"b2z-stack-{args.arm}-"))
    struct_dir = work / "out"; struct_dir.mkdir(parents=True)
    msa_dir = work / "msa"; msa_dir.mkdir(parents=True)
    target = FIX / f"{args.fixture}.yaml"
    _seed_msa(target, (FIX / f"{args.fixture}.a3m").read_text(), msa_dir)
    cfg = build_cfg(msa_dir, struct_dir, args.model)
    _ensure_local_artifacts(cfg)

    t_load = time.perf_counter()
    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("b2z-ttnn-upgrade", cfg)
    model_load_s = round(time.perf_counter() - t_load, 3)

    env = {
        "arm": args.arm, "round": args.round, "ttnn": ttnn_version,
        "torch": torch.__version__, "host": socket.gethostname(),
        "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
        "grid": grid, "arch": str(getattr(dev, "arch", lambda: "?")()),
        "python": sys.executable, "tt_bio_file": _TB.__file__, "model": args.model,
        # two arms can report the same ttnn VERSION and still be different builds: the
        # profiler build is v0.68.0 from source, the wheel is v0.68.0 from pip. The file
        # is what tells them apart.
        "ttnn_file": ttnn.__file__, "tt_metal_home": os.environ.get("TT_METAL_HOME"),
        "model_load_s": model_load_s, "fixture": args.fixture,
        "protocol": {"recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                     "diffusion_samples": DIFFUSION_SAMPLES, "seed": SEED},
    }

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    def emit(rec: dict) -> None:
        with out.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    def fold(kind: str, idx: int) -> dict:
        for p in struct_dir.glob("*"):
            p.unlink() if p.is_file() else shutil.rmtree(p)
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        metrics, _b, _f = state.predict_one(target, cfg)
        ttnn.synchronize_device(dev)
        wall = time.perf_counter() - t0
        cifs = sorted(struct_dir.glob("*.cif"))
        assert cifs, "no CIF written"
        blob = cifs[0].read_bytes()
        if args.cifdir:
            args.cifdir.mkdir(parents=True, exist_ok=True)
            (args.cifdir / f"{args.arm}_r{args.round}_{idx}.cif").write_bytes(blob)
        rec = dict(env)
        rec.update({
            "kind": kind, "i": idx,
            "fold_s": round(wall, 4),
            "cif_sha256": hashlib.sha256(blob).hexdigest(),
            "plddt": float(metrics.get("plddt", float("nan"))) if isinstance(metrics, dict) else None,
            "loadavg": os.getloadavg(),
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        })
        emit(rec)
        return rec

    r = fold("cold", 0)
    print(f"[{args.arm} r{args.round}] cold {r['fold_s']:.3f}s ttnn={ttnn_version}", flush=True)
    for i in range(1, args.folds + 1):
        r = fold("warm", i)
        print(f"[{args.arm} r{args.round}] warm{i} {r['fold_s']:.3f}s "
              f"cif={r['cif_sha256'][:16]}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
