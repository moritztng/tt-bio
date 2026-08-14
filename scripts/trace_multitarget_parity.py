#!/usr/bin/env python3
"""Multi-target trace-replay parity gate (regression test for the shape-keyed
trace corruption bug).

``--trace`` replays a captured ttnn trace of the per-step denoise device
stream. The captured graph closes over the DEVICE BUFFERS of the conditioning
it was recorded with, but the trace cache was keyed on shape alone (``N`` in
``protenix.py:denoise_traced``, ``(B, N_padded)`` in
``tenstorrent.py:forward_traced``) while every fold stages its conditioning
into fresh device buffers. A second same-shape target in one process therefore
replayed the FIRST target's conditioning and silently returned a wrong
structure. Root-caused 2026-08-14 on OpenDDE 512 aa: fold 1 of a trace-ON
process correct, every later fold deterministically wrong (plDDT 0.754 ->
0.617). A one-fold-per-process parity check cannot see this bug class; this
gate exists so there is always a second target in the process.

The gate folds two DIFFERENT targets with the SAME atom count in one process
(bk6_104, 9BK6 chain A, and a circular rotation of it: identical composition
keeps the atom count and trace bucket identical, the rotated sequence is a
different target with different conditioning), trace OFF then trace ON, and
requires each target's CIF bytes under trace to equal its own untraced bytes.

--model protenix-v2 / opendde exercise the protenix.py denoise_traced trace
(the buggy one, fixed 2026-08-14). --model boltz2 exercises the
tenstorrent.py forward_traced trace (shared with BoltzGen): that path is
reset per fold by Boltz2.predict_step, so it was never exposed to the bug;
the gate guards it as a regression check on the identity-guard change.

    TT_VISIBLE_DEVICES=1 python3 scripts/trace_multitarget_parity.py --model opendde
    TT_VISIBLE_DEVICES=1 python3 scripts/trace_multitarget_parity.py --model protenix-v2
    TT_VISIBLE_DEVICES=1 python3 scripts/trace_multitarget_parity.py --model boltz2

Exit 0 = parity holds; exit 1 = a fold under trace diverged from its untraced
reference (the bug is back). Production config throughout: 10 recycling steps,
200 sampling steps, 1 sample, seed 0, MSA on (seeded into the cache, no search).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# The trace region must be reserved at device open; get_device reads this env
# var then. Set before any tt_bio import that could open the device.
os.environ.setdefault("TT_BIO_TRACE_REGION_SIZE", str(1 << 30))

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "scripts" / "gpu_vs_tt" / "fixtures" / "distinct"

# Production config, identical to scripts/gpu_vs_tt/tt_baseline.py.
RECYCLING_STEPS = 10
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 1
SEED = 0
ROT = 37  # circular shift for the second target; any k in [1, L) works


def _rotate(s: str, k: int) -> str:
    return s[k:] + s[:k]


def _rotate_a3m(text: str, k: int) -> str:
    """Circular-shift every alignment row by the same k. Keeps columns aligned
    and the query row (row 1) equal to the rotated target sequence, so the
    harness's query-row contract holds for the rotated target too."""
    out = []
    for line in text.split("\n"):
        if line and not line.startswith(">"):
            line = _rotate(line, k % len(line))
        out.append(line)
    return "\n".join(out)


def _write_target(work: Path, name: str, seq: str) -> Path:
    y = work / f"{name}.yaml"
    y.write_text(
        "version: 1\nsequences:\n  - protein:\n"
        f"      id: A\n      sequence: {seq}\n"
    )
    return y


def _seed_msa(target: Path, a3m_text: str, msa_dir: Path) -> None:
    """Install ``a3m_text`` as the cached alignment for ``target``'s single
    chain ({sha256(seq)[:16]}.a3m), so no MSA search ever runs. Same contract
    as tt_baseline.seed_msa_cache, including the query-row assertion."""
    from tt_bio.main import _read_bio_chains

    chains = _read_bio_chains(target)
    assert len(chains) == 1, f"{target} is not a monomer: {len(chains)} chains"
    seq = chains[0][1]
    rows = a3m_text.split("\n")
    assert rows[1] == seq, f"a3m query row does not match {target}'s sequence"
    msa_dir.mkdir(parents=True, exist_ok=True)
    h = hashlib.sha256(seq.encode()).hexdigest()[:16]
    (msa_dir / f"{h}.a3m").write_text(a3m_text)


def _diffusion_module(model: str, state):
    """The DiffusionModule holding the trace, for the trace-engaged check."""
    if model == "opendde":
        return state.model._protenix.diffusion
    if model == "boltz2":
        return state.model.structure_module.score_model
    return state.model.diffusion


def _trace_bucket(model: str, state):
    """The shape key of the live captured trace, or None if no trace is held.
    protenix.py keys its trace on atom count N; tenstorrent.py on
    (B, N_padded)."""
    dm = _diffusion_module(model, state)
    tr = getattr(dm, "_trace", None) or getattr(dm, "_diff_trace", None)
    if not tr:
        return None
    return tr.get("N", tr.get("N_padded"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True,
                    choices=["protenix-v2", "opendde", "boltz2"])
    ap.add_argument("--out", type=Path, default=None,
                    help="optional JSON dump of the per-fold digests")
    args = ap.parse_args()

    import torch  # noqa: F401
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E

    _noop = lambda *a, **k: None
    _E.set_progress(_noop)

    get_device()  # open the chip once, with the trace region (env set above)

    work = Path(tempfile.mkdtemp(prefix=f"trace-parity-{args.model}-"))
    struct_dir = work / "out"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir = work / "msa"
    msa_dir.mkdir(parents=True, exist_ok=True)

    seq_a = (FIXTURES / "bk6_104.seq").read_text().split()[-1].strip()
    seq_b = _rotate(seq_a, ROT)
    assert sorted(seq_a) == sorted(seq_b) and seq_a != seq_b
    a3m_a = (FIXTURES / "bk6_104.a3m").read_text()
    a3m_b = _rotate_a3m(a3m_a, ROT)
    tgt_a = _write_target(work, "bk6_104", seq_a)
    tgt_b = _write_target(work, "bk6_104_rot37", seq_b)
    _seed_msa(tgt_a, a3m_a, msa_dir)
    _seed_msa(tgt_b, a3m_b, msa_dir)

    cfg = dict(
        model=args.model, fast=False, output_format="cif",
        recycling_steps=RECYCLING_STEPS, sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES, seed=SEED, trace=False,
        msa_dir=str(msa_dir), struct_dir=str(struct_dir),
        use_msa_server=True, msa_db_path=None, use_envdb=False, msa_endpoint=None,
        single_sequence=False, msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy", msa_server_username=None,
        msa_server_password=None, api_key_value=None, max_msa_seqs=8192,
        write_pae=False, write_pde=False, write_embeddings=False, method=None,
    )
    if args.model == "boltz2":
        # Production conf_kwargs from main.py's predict CLI. diffusion_trace is
        # a Boltz2 load-time kwarg (reserves the trace region up front); the
        # per-fold A/B toggles AtomDiffusion._diffusion_trace, which
        # preconditioned_network_forward reads on every call.
        cfg["conf_kwargs"] = dict(
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
            pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0,
                             "v2": True},
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
            diffusion_trace=True,
        )
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    state.load_model(cfg)
    state.bind_run("trace-parity", cfg)
    state.pfn = _noop

    def one_fold(target: Path, trace: bool):
        job_cfg = dict(cfg, trace=trace, struct_dir=str(struct_dir))
        if args.model == "boltz2":
            state.model.structure_module._diffusion_trace = trace
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(target, job_cfg)
        dt = time.perf_counter() - t0
        cifs = {f.name: hashlib.sha256(f.read_bytes()).hexdigest()[:16]
                for f in sorted(struct_dir.glob("*.cif"))}
        assert cifs, f"no CIF written for {target.name}"
        plddt = metrics.get("plddt")
        if plddt is None:
            plddt = metrics.get("complex_plddt")  # boltz2 names it this way
        return {"target": target.stem, "trace": trace, "s": round(dt, 3),
                "plddt": plddt, "cif_sha256": cifs,
                "trace_N": _trace_bucket(args.model, state)}

    # OFF A/B establish each target's untraced reference; ON A captures the
    # trace; ON B is the replay that must still return B's own structure;
    # the final OFF A is the process-level determinism control.
    plan = [(tgt_a, False), (tgt_b, False), (tgt_a, True), (tgt_b, True),
            (tgt_a, False)]
    folds = [one_fold(t, tr) for t, tr in plan]

    def sha(rec):
        return next(iter(rec["cif_sha256"].values()))

    off_a, off_b, on_a, on_b, off_a2 = folds
    print(f"\n{'fold':<22}{'trace':<7}{'s':>9}  {'plDDT':>8}  CIF sha256")
    for r in folds:
        print(f"{r['target']:<22}{str(r['trace']):<7}{r['s']:>9.3f}  "
              f"{(r['plddt'] or 0):>8.4f}  {sha(r)}")

    problems = []
    if on_a["trace_N"] is None or on_b["trace_N"] is None:
        problems.append("trace did not engage (no captured trace after ON folds)")
    elif on_a["trace_N"] != on_b["trace_N"]:
        problems.append(f"trace N differs across targets ({on_a['trace_N']} vs "
                        f"{on_b['trace_N']}): the two targets do not share a "
                        "trace bucket, the repro is vacuous")
    if sha(off_a) == sha(off_b):
        problems.append("the two targets folded to identical CIFs untraced; "
                        "they are not actually distinct, the repro is vacuous")
    if sha(off_a) != sha(off_a2):
        problems.append(f"process-level nondeterminism: untraced A gave "
                        f"{sha(off_a)} then {sha(off_a2)}")
    if sha(on_a) != sha(off_a):
        problems.append(f"trace-ON A diverged from untraced A: {sha(on_a)} vs "
                        f"{sha(off_a)} (first/capture fold must be exact)")
    if sha(on_b) != sha(off_b):
        problems.append(f"trace-ON B diverged from untraced B: {sha(on_b)} vs "
                        f"{sha(off_b)} -- the second target replayed stale "
                        "conditioning (the multi-target trace bug)")

    if args.out:
        args.out.write_text(json.dumps(
            dict(model=args.model, git=_git_sha(), folds=folds,
                 problems=problems), indent=2) + "\n")

    if problems:
        print("\nFAIL:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nPASS: trace-ON folds are byte-identical to their untraced "
          "references for both targets in one process.")
    return 0


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
