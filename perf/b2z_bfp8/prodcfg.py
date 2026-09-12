#!/usr/bin/env python3
"""The production Boltz-2 worker config and device, built once for every harness in this bet.

Both of this bet's harnesses crashed on 2026-09-11 because each built its own idea of the
config and the device: the screen handed `_WorkerState.load_model` a dict with no
`conf_kwargs`, and the fold A/B opened a device with no trace region under a `diffusion_trace`
config, so the denoiser's traced forward had nowhere to put the trace. Both are the same bug,
so there is one fix: the shipped `tt-bio predict` CLI (`tt_bio/main.py:3155-3205`) is the only
authority on what that config contains, and this module mirrors it. A number measured through
any other config is not the published cell's number.

Importing this module also reserves the trace region, because `get_device` reads
`TT_BIO_TRACE_REGION_SIZE` at open and the open happens on first touch.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

# Read by get_device at open, so it has to be set before anything touches the device.
#
# 512 MiB, not the 1 GiB this file shipped with. `tt_bio.get_device()` with >= 1 GiB hangs forever
# at open on whglx's 32-chip Wormhole mesh and leaves the chip half-initialised -- tt-bio's open
# plus the big region plus 32 chips, none of the three alone (`b2z-bfp8-narrow` pass 5, which
# root-caused it after mistaking the hung chips for dead silicon). The threshold is between
# 512 MiB and 1 GiB; a p300c is unaffected but does not need the extra region either.
os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(512 << 20))

REPO = Path(__file__).resolve().parents[2]
FIX = REPO / "perf" / "size512" / "fixtures"

#: the published 512 aa cell's protocol (state/boltz2-2x/PLAN.md)
RECYCLES, STEPS, SAMPLES, SEED = 3, 200, 1, 0

# Verbatim from tt_bio/main.py's predict CLI. Kept as literals rather than imported because
# that builder lives inside a click command and takes 40 CLI options.
_DIFFUSION = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
              "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
              "P_mean": -1.2, "P_std": 1.5, "coordinate_augmentation": True,
              "alignment_reverse_diff": True, "synchronize_sigmas": True}
_PAIRFORMER = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
_MSA = {"subsample_msa": False, "num_subsampled_msa": 1024, "use_paired_feature": True,
        "msa_s": 64, "msa_blocks": 4, "msa_dropout": 0.15, "z_dropout": 0.25,
        "pairwise_head_width": 32, "pairwise_num_heads": 4, "activation_checkpointing": True}


def build_cfg(msa_dir: Path, struct_dir: Path, *, steps: int = STEPS,
              recycles: int = RECYCLES, samples: int = SAMPLES, seed: int = SEED) -> dict:
    """The worker config `tt-bio predict --model boltz2` builds, with artifacts resolved."""
    from tt_bio.worker import _ensure_local_artifacts

    cfg = dict(
        model="boltz2", fast=False, output_format="cif",
        recycling_steps=recycles, sampling_steps=steps, diffusion_samples=samples,
        seed=seed, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=False, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None, msa_server_password=None,
        api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
        conf_kwargs=dict(
            predict_args={"recycling_steps": recycles, "sampling_steps": steps,
                          "diffusion_samples": samples, "max_parallel_samples": 5},
            diffusion_process_args=_DIFFUSION, pairformer_args=_PAIRFORMER, msa_args=_MSA,
            steering_args={"fk_steering": False, "physical_guidance_update": False,
                           "contact_guidance_update": True, "num_particles": 3,
                           "fk_lambda": 4.0, "fk_resampling_interval": 3, "num_gd_steps": 20},
            use_kernels=True, use_tenstorrent=True, trace=False, diffusion_trace=True),
    )
    _ensure_local_artifacts(cfg)
    return cfg


def seed_msa(target: Path, a3m_text: str, msa_dir: Path) -> None:
    """Drop the fixture's own a3m where the featurizer looks for it, keyed by sequence hash."""
    from tt_bio.main import _read_bio_chains

    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    assert a3m_text.split("\n")[1] == seq, "a3m query row does not match the target sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    (msa_dir / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m").write_text(a3m_text)


def assert_checkout() -> str:
    """Fail loudly if this ran the SHARED checkout instead of this worktree.

    The venv installs `tt_bio` from /home/ttuser/tt-bio-dev, so an import that does not resolve
    to REPO scores a branch that is not this one (memory
    `parity-gate-scores-installed-package-not-checkout`).
    """
    import tt_bio

    got = Path(tt_bio.__file__).resolve()
    assert str(got).startswith(str(REPO) + "/"), f"imported {got}, not the worktree at {REPO}"
    return str(got)
