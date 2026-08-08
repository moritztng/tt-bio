#!/usr/bin/env python3
"""Standing accuracy release gate — the on-hardware accuracy leg of RELEASING.md.

For every shipped fold architecture (Boltz-2, ESMFold2, ESMFold2-fast,
Protenix-v2, OpenFold3, OpenDDE) this
folds one easy, foldable target end-to-end on the real device with production
sampling and then applies two independent gates to the result:

  1. PARSE   — the written mmCIF must load under a strict ``Bio.PDB.MMCIFParser``.
               Biopython is stricter about required ``_atom_site`` columns than the
               geometry parser below, so it is the right tool to catch writer/format
               regressions (e.g. the missing-occupancy bug fixed in 17aeab9e).
  2. RMSD/TM — the confidence-selected structure (best-of-N, exactly what a user
               receives) must land within a per-model ground-truth CA-RMSD / TM-score
               floor of the experimental structure. Reuses the Kabsch + TM + best-of-N
               harness in ``tests/test_structure.py`` (do not re-derive it here).

Self-consistency (seed-vs-reference RMSD) is NOT sufficient — it passes even when the
fold is wrong. A tag must clear a real
ground-truth floor for every model.

Each model folds here the way it is actually used: the MSA-dependent models
(``tt_bio.main.MSA_DEFAULT_MODELS``) get an MSA, esmfold2 and esmfold2-fast fold
single-sequence — see ``_msa_args``.

BoltzGen is a *design* model, not a fold model — there is no ground truth to fold
against, so it is gated separately from the four above. Its correctness bar is
designability (self-consistency RMSD, scRMSD): refold each design's sequence in
isolation with Boltz-2 and check the shape reproduces. This is the exact
``scripts/boltzgen_designability.py`` method already validated on this hardware
(docs/boltzgen-designability.md; also confirmed at n=8) — reused here, not re-derived. At n=4 (production 500-step sampling) a full
design+refold+analysis run measured ~4.5 min on Blackhole, comparable to a fold
model's leg, so it runs by default alongside the other four rather than standalone
(supersedes docs/boltzgen-designability.md's earlier "keep it out of the fast gate"
call, which assumed a much slower per-design cost).

ESMC is an *embedding* model, not a fold model — it has no structure to score
against ground truth, so the RMSD/TM mechanism above does not apply. Its
correctness bar is embedding-space agreement with the reference esm ESMC: the
shipped embed path (``tt_bio.esmc.load_esmc`` + ``embed_sequences``) must match
the reference's per-residue embeddings at PCC >= 0.99 on a real protein. This is
the gate that the fused-RoPE numerics change (``esmc._rope`` →
``ttnn.experimental.rotary_embedding``) was held on — the bucketed embed path
always takes the fused kernel (``BUCKET=64`` pads L tile-aligned), so this leg
exercises it directly. Reuses ``scripts/esmc_embed_parity.py``'s
``run_esmc_parity`` (and the ``tests/esmc_reference.py`` golden) — not re-derived
here. 300m/600m (the embed workhorses) run by default; esmc-6b is opt-in
(``--model esmc-6b``) since its ~13 GB load dominates wall-clock and is too slow
for the fast gate. It is *not* opt-in for accuracy reasons: the 6b leg has been
run on-device against the esm-repo fp32 reference (same golden as 300m/600m, at
the 6b config) on the four benchmark proteins and passes at the same bar
(per-residue embedding PCC 0.99904–0.99969, device self-consistency 1.00000 —
see docs/implementation-parity.md's ESMC-6b row). The 6b port is the same code path
as 300m/600m (same Block/Embedding modules, same fused RoPE, head_dim 64), so
300m/600m's default-gate parity is a cheap proxy; the opt-in 6b leg is the
standing on-device confirmation. ``run_esmc_parity`` delegates to
``scripts/esmc6b_embed_parity.py`` for 6b (sharded TE safetensors, no sequence
head — the single-.pth / logits path does not apply).

OpenDDE-abag is an antibody-antigen *docking* leg, not a single-chain fold — the
opendde-abag checkpoint co-folds a Fab + antigen complex, so the correctness bar is
docking accuracy (global DockQ of the confidence-selected complex vs the experimental
complex), NOT the CA-RMSD/TM floor the MODELS dict applies to 7ROA. It therefore runs
as its own leg (same shape as the BoltzGen designability leg), reusing the 1AHW fixture
committed by the prior parity cross-check (``examples/1ahw_abag.yaml`` +
``examples/ground_truth_structures/1ahw.cif`` — public PDB 1ahw, ASU chains A/B/C) and
the reference DockQ tool (``scripts/opendde_dockq.py``, DockQ==2.1.3, the Wallner-lab
implementation that defines the metric) — neither re-derived here. DockQ is an
eval-time requirement, not a project runtime dep, so the gate scores this leg by
shelling out to ``OPENDDE_DOCKQ_PYTHON`` (defaults to the gate's own python; set it to
a venv that has DockQ installed if the gate venv does not).

Capacity is a separate leg because the legs above cannot see it. They compare NUMBERS, so
a change that grows the device-memory footprint either still fits (identical numbers, PASS)
or dies with an allocation error — nothing in a parity fixture reports "this now needs 10 GB
more". That is how multiplicity batching shipped verified on small inputs and then ran out of
memory on real antibody-antigen targets. The capacity leg folds the LARGEST supported target
(``examples/abag_pilot_expansion/9j4c_abag.yaml``, 1095 tokens) at the largest sample count
the campaigns use, measures the peak device DRAM via ``TT_BIO_DRAM_PEAK``, and checks it
against a budget — plus the per-sample output contract, since a fold that quietly writes
fewer structures than it was asked for is the other face of the same failure. Six sampling
steps and single-sequence on purpose; see the constants for why.

    # gate everything (five fold models + BoltzGen designability + ESMC embed parity
    # + OpenDDE-abag docking) on card 1
    TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> ESM_ROOT=/path/to/esm \
        OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python \
        python scripts/release_gate.py
    # one leg
    python scripts/release_gate.py --model protenix-v2
    python scripts/release_gate.py --model boltzgen
    python scripts/release_gate.py --model esmc-300m
    python scripts/release_gate.py --model opendde-abag
    python scripts/release_gate.py --model capacity

Exit code 0 iff every requested model PASSES its gate; 1 otherwise. Runs on the
device serially (one card context per run); no CPU shortcut for the fold/design.

This is the *accuracy* leg of the release gate. The *UX* leg lives in
``scripts/ux_regression.py`` (live-progress phases, output parsing, CLI shape) —
see RELEASING.md. The two are independent; both must exit 0 before a tag.
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The default foldable gate target: examples/prot.yaml == PDB 7ROA, a 117-residue
# monomer that Boltz-2 folds to 1.55 A — proof the target is easy, so a large RMSD
# is a real model/port problem, not a hard target.
DATA = REPO_ROOT / "examples" / "prot.yaml"
GROUND_TRUTH = REPO_ROOT / "examples" / "ground_truth_structures" / "prot.cif"
NAME = DATA.stem  # "prot" -> results land in <model>_results_prot/

# Production sampling. n_step=10 undersamples and can fail a correct model; 200 steps / 5
# samples is the floor for a real accuracy read.
SAMPLING_STEPS = 200
DIFFUSION_SAMPLES = 5
SEED = 0
# When set (via --fast), fold with tt-bio --fast so the gate exercises the
# block-fp8 trunk path (bf8 weights + bf8 matmul output) that ships under --fast.
# Defaults off: the standing floors below were calibrated for full precision.
FAST = False
# When set (via --diffusion_trace), fold boltz2 with the per-step DiT trace
# replay on (lossless; reserves a 1 GiB trace region). boltz2 only — other fold
# models do not wire diffusion_trace through. Defaults off.
DIFFUSION_TRACE = False

# Per-model ground-truth floors on 7ROA, of the confidence-selected structure.
# Anchored to measured on-hardware baselines with margin for TT diffusion's
# seed-to-seed stochasticity — deliberately generous
# floors that catch a regression or a gross fold failure, NOT tight targets. Tighten as
# a model's baseline distribution is nailed down; never set below what a correct fold hits.
#   measured best-conf: Boltz-2 1.55 A / TM 0.93 | Protenix-v2 3.87 / 0.71
# ESMFold2's floor is anchored to its DEFAULT single-sequence fold, measured 2026-07-26 on
# Blackhole: 5.80 A / TM 0.508 (esmfold2-fast, same target, same run: 1.73 A / TM 0.909). The old
# 4.0/0.65 came from an MSA-on fold, which is not what a user gets by default — see _msa_args. That
# ESMFold2 needs the MSA on 7ROA while its lighter checkpoint does not is a model-quality property,
# not a port defect: the single-sequence path is parity-verified against the torch reference at
# 0.14-0.75 A on trp-cage/GB1/ubiquitin/lysozyme (docs/implementation-parity.md). The cost is a
# loose floor for this one model here; tight ESMFold2 numerics live in full_parity_gate.py's
# esmfold2 leg. To tighten it, move esmfold2's gate target to one of those four (needs a per-model
# target in this gate, which currently folds one shared DATA for all five).
MODELS = {
    "boltz2":        {"max_rmsd": 3.0, "min_tm": 0.75},
    "esmfold2":      {"max_rmsd": 8.0, "min_tm": 0.40},
    "esmfold2-fast": {"max_rmsd": 4.5, "min_tm": 0.60},
    "protenix-v2":   {"max_rmsd": 6.0, "min_tm": 0.50},
    "opendde":       {"max_rmsd": 6.0, "min_tm": 0.50},
    # OpenFold3 (polymer-only port; protenix-v2 is the 1:1 architectural analogue).
    # Measured on this gate 2026-08-07 (qb2 p150a, MSA on): RMSD 1.775 A / TM 0.890.
    # Floor = ~2x measured, same discipline as boltz2 (1.55 -> 3.0) and protenix-v2
    # (3.87 -> 6.0): catches a gross failure, not run-to-run MSA-draw noise.
    "openfold3":     {"max_rmsd": 3.5, "min_tm": 0.70},
}

# BoltzGen designability leg — see module docstring. Small n and the target the
# README already documents for `tt-bio design --model boltzgen`; kept fast enough for a
# release gate
# while still statistically meaningful (docs/boltzgen-designability.md's n=4 run on
# this exact target/protocol measured 1.00 A median / 75% <=2A; a fresh n=4
# reproduction on 2026-07-10 main HEAD measured 0.85 A median / 100% <=2A in 271s).
# Strict 2 A bar (BoltzGen's own designable threshold) with a generous 50% pass-rate
# floor — same "catch a gross failure, not a tight target" philosophy as the MODELS
# floors above: one bad seed out of four should not fail the gate, all four should.
BOLTZGEN_SPEC = REPO_ROOT / "examples" / "binder.yaml"
BOLTZGEN_PROTOCOL = "protein-anything"
BOLTZGEN_NUM_DESIGNS = 4
BOLTZGEN_SC_THRESHOLD = 2.0
BOLTZGEN_MIN_PASS_RATE = 0.5

# ESMC embedding-parity leg — see module docstring. Per-residue embedding PCC
# floor vs the reference esm ESMC on a real protein. Generous (the shipped fused
# path measures ~0.9996-0.9998): catches a gross numerics regression, not a tight
# target, same philosophy as the fold floors. 300m/600m are the embed workhorses
# (`tt-bio embed`, JapanFold embeddings) and run by default; 6b is opt-in — it
# passes the same bar on-device (see docs/implementation-parity.md's ESMC-6b row) but
# its ~13 GB load is too slow for the fast gate, so 300m/600m cover it by default.
ESMC_MIN_PCC = 0.99
ESMC_DEFAULT = ["esmc-300m", "esmc-600m"]
ESMC_OPT_IN = ["esmc-6b"]

# OpenDDE-abag antibody-antigen docking leg — see module docstring. The opendde-abag
# checkpoint co-folds a Fab + antigen complex, so the correctness bar is docking
# accuracy (global DockQ of the confidence-selected complex vs the experimental
# complex), NOT the single-chain CA-RMSD/TM floor the MODELS dict applies to 7ROA. It
# therefore runs as its own leg (same shape as the BoltzGen designability leg),
# reusing the 1AHW fixture committed by the prior parity cross-check
# (examples/1ahw_abag.yaml + examples/ground_truth_structures/1ahw.cif — public PDB
# 1ahw, ASU chains A/B/C) and the reference DockQ tool (scripts/opendde_dockq.py,
# DockQ==2.1.3, the Wallner-lab implementation that defines the metric) — neither
# re-derived here. Measured baseline on a Blackhole card: global DockQ 0.863
# best-confidence / 0.882 oracle (best-of-5, MSA on, 10 recycles / 200 steps); the
# floor catches a gross mis-dock (the 9dsg-class failure mode scores 0.011 / fnat 0),
# not a tight target — same philosophy as the MODELS floors.
OPENDDE_ABAG_DATA = REPO_ROOT / "examples" / "1ahw_abag.yaml"
OPENDDE_ABAG_NATIVE = REPO_ROOT / "examples" / "ground_truth_structures" / "1ahw.cif"
OPENDDE_ABAG_MIN_DOCKQ = 0.50
# DockQ is an eval-time requirement (not a project runtime dep), installed in a
# separate venv on the release host. The gate scores this leg by shelling out to this
# python running scripts/opendde_dockq.py; defaults to the gate's own python so a host
# that carries DockQ in the gate venv needs no extra config, and a host that does not
# sets OPENDDE_DOCKQ_PYTHON to a venv that does (mirrors the ESMC leg's ESM_ROOT).
OPENDDE_DOCKQ_PYTHON = os.environ.get("OPENDDE_DOCKQ_PYTHON", sys.executable)

# Every fold this gate launches is bounded by a hard wall-clock timeout so a flaky external
# MSA server (the v0.3.3 release lost ~25 min to a ColabFold hang) or a hung multiprocessing
# shutdown can never block the gate forever. The offline fallback for when the public ColabFold
# service is down/flaky: set RELEASE_GATE_MSA_DIR to a dir holding the cached
# `{sha256(sequence)[:16]}.a3m` files (see RELEASING.md); the fold then runs with --msa_dir and
# never touches the network. Both are env-tunable so a slow host can raise the timeout.
FOLD_TIMEOUT_S = int(os.environ.get("RELEASE_GATE_FOLD_TIMEOUT", "1800"))
MSA_DIR = os.environ.get("RELEASE_GATE_MSA_DIR")

# --- capacity leg ----------------------------------------------------------------------
# The accuracy legs above compare NUMBERS, so a change that grows the device-memory
# footprint is invisible to them: the fold either still fits (identical numbers, PASS) or
# dies with an allocation error the harness reports as a per-item status. That is how
# multiplicity batching shipped verified-on-small-inputs and then ran out of memory on real
# antibody-antigen targets. So the gate also measures the footprint directly, at the
# LARGEST supported input and the LARGEST sample count the campaigns use.
#
# Single-sequence on purpose: the sample-scaled footprint is what a batching change moves,
# and folding single-sequence keeps the budget reproducible across hosts instead of drifting
# with MSA depth and MSA-server state. Six sampling steps on purpose: peak DRAM is a
# per-step high-water mark, not something that accumulates over the trajectory, so the
# measurement is step-count-independent and there is no reason to pay for 200.
CAPACITY_STEPS = 6
# Budgets, in GiB of device DRAM. Leg 1's measured peak on a Blackhole p150a is 5.90 GiB,
# so 7.0 leaves ~19% headroom: loose enough that it does not chase run-to-run noise (the
# shapes are deterministic, so there is very little), tight enough to catch the small
# regressions too -- re-replicating just the atom-transformer pair bias and the windowed
# atom pair tensor would add ~0.9 GiB, and re-replicating the DiT pair biases would add
# ~9.6 GiB. Env-tunable so a card with a different budget can gate against its own.
#
# Leg 2 is the structural-token ceiling: opendde-abag's refiner runs the pair track at
# ~1.9x the residue count, which is where the WH 12 GiB OOM exclusions came from. Fewer
# samples than leg 1 because the pair-track peak this leg guards is sample-count
# independent; the sample-scaled diffusion footprint is leg 1's job.
CAPACITY_LEGS = [
    # (yaml, model, residue tokens, samples, mps, budget GiB)
    ("examples/abag_pilot_expansion/9j4c_abag.yaml", "protenix-v2", 1095, 50, 5,
     float(os.environ.get("RELEASE_GATE_CAPACITY_MAX_GIB", "7.0"))),
    ("examples/abag_xm/9ivj.yaml", "opendde-abag", 891, 8, 2,
     float(os.environ.get("RELEASE_GATE_CAPACITY_MAX_GIB_2", "12.0"))),
]


def _msa_args(model: str) -> list:
    """MSA source for one model's gate fold — the way that model is ACTUALLY used.

    ``tt_bio.main.MSA_DEFAULT_MODELS`` is the source of truth for which models resolve an MSA
    by default (boltz2 / protenix-v2 / opendde / opendde-abag): those fold with an offline
    cached-a3m dir if RELEASE_GATE_MSA_DIR is set, otherwise the ColabFold server (bounded by
    FOLD_TIMEOUT_S — see RELEASING.md). esmfold2 is single-sequence with an optional MSA and
    esmfold2-fast ships no MSA encoder at all, so both fold single-sequence here: gating a
    config no user reaches by default leaves the default path untested. esmfold2's optional
    MSA-conditioned trunk keeps its own coverage in tests/test_esmfold2.py::test_msa_encoder
    (on-device MSAEncoder parity, run by RELEASING.md's pytest step)."""
    from tt_bio.main import MSA_DEFAULT_MODELS
    if model not in MSA_DEFAULT_MODELS:
        return []
    return ["--msa_dir", MSA_DIR] if MSA_DIR else ["--use_msa_server"]


def _run_fold(cmd: list, timeout: float, **popen_kw) -> tuple:
    """Run a fold subprocess in its OWN process group; on timeout kill the whole group so a
    hung MSA-server wait or a hung multiprocessing shutdown cannot orphan device-holding
    children (which would wedge the card for later legs). Returns (returncode, timed_out)."""
    import signal
    proc = subprocess.Popen(cmd, start_new_session=True, **popen_kw)
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
                proc.wait(timeout=10)
                break
            except Exception:
                continue
        return None, True


def _load_structure_harness():
    """Import tests/test_structure.py by path (tests/ is not an installed package)."""
    path = REPO_ROOT / "tests" / "test_structure.py"
    spec = importlib.util.spec_from_file_location("tt_bio_test_structure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_designability_harness():
    """Import scripts/boltzgen_designability.py by path — reuse its _run_design/score,
    do not re-derive the design-pipeline invocation or the scRMSD harvest."""
    path = REPO_ROOT / "scripts" / "boltzgen_designability.py"
    spec = importlib.util.spec_from_file_location("tt_bio_boltzgen_designability", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_esmc_parity_harness():
    """Import scripts/esmc_embed_parity.py by path — reuse its run_esmc_parity +
    tests/esmc_reference.py golden; do not re-derive the ESMC parity harness."""
    path = REPO_ROOT / "scripts" / "esmc_embed_parity.py"
    spec = importlib.util.spec_from_file_location("tt_bio_esmc_embed_parity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _results_cifs(results_dir: Path) -> list[Path]:
    d = results_dir / "structures"
    return sorted(d.glob(f"{NAME}*.cif")) if d.exists() else []


def _parse_gate(cifs: list[Path], name: str = NAME) -> None:
    """Strict Bio.PDB.MMCIFParser parse of every written sample. Raises on a bad file."""
    from Bio.PDB import MMCIFParser
    from Bio.PDB.PDBExceptions import PDBConstructionWarning

    if not cifs:
        raise FileNotFoundError("predict wrote no CIF output")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PDBConstructionWarning)  # promote writer sloppiness to a failure
        for cif in cifs:
            structure = MMCIFParser(QUIET=True).get_structure(name, str(cif))
            n_atoms = sum(1 for _ in structure.get_atoms())
            if n_atoms == 0:
                raise ValueError(f"{cif.name}: parsed but contains 0 atoms")


def run_model(model: str, harness, keep: bool) -> dict:
    """Fold, parse, and ground-truth-score one model. Returns a result row."""
    from tt_bio.main import predict_results_dir_name
    out = REPO_ROOT / predict_results_dir_name(model, NAME)
    if out.exists():
        shutil.rmtree(out)  # never score a stale run if this predict crashes

    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(DATA),
        "--model", model,
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--seed", str(SEED),
        *_msa_args(model),
        "--out_dir", str(REPO_ROOT),
    ] + ((["--fast"] if FAST else [])
          + (["--diffusion_trace"] if (DIFFUSION_TRACE and model == "boltz2") else []))
    print(f"\n{'='*70}\n[{model}] folding {DATA.name} "
          f"({SAMPLING_STEPS} steps, {DIFFUSION_SAMPLES} samples)\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "rmsd": None, "tm": None,
           "parse": False, "gate": False, "error": None}
    t0 = time.monotonic()
    rc, timed_out = _run_fold(cmd, FOLD_TIMEOUT_S, cwd=REPO_ROOT)
    row["seconds"] = time.monotonic() - t0
    if timed_out:
        row["error"] = (f"predict timed out after {FOLD_TIMEOUT_S}s (flaky MSA server? set "
                        f"RELEASE_GATE_MSA_DIR to a dir with the cached a3m — see RELEASING.md)")
        return row
    if rc != 0:
        row["error"] = f"predict exited {rc}"
        return row

    cifs = _results_cifs(out)
    try:
        _parse_gate(cifs)
        row["parse"] = True
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row

    # Ground-truth RMSD/TM of the confidence-selected structure (harness reads
    # <model>_results_<NAME>/ and examples/ground_truth_structures/ relative to REPO_ROOT).
    try:
        rmsd, tm = harness.evaluate(NAME, results_dir=out)  # no thresholds -> returns numbers, never raises
    except Exception as e:
        row["error"] = f"RMSD eval failed: {e}"
        return row
    row["rmsd"], row["tm"] = rmsd, tm

    th = MODELS[model]
    row["gate"] = (rmsd <= th["max_rmsd"]) and (tm >= th["min_tm"])

    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def run_boltzgen(bg, keep: bool) -> dict:
    """Design, parse, and designability-score BoltzGen. Returns a result row."""
    out = REPO_ROOT / "boltzgen_gate_binder"
    if out.exists():
        shutil.rmtree(out)  # never score a stale run if this design run crashes

    print(f"\n{'='*70}\n[boltzgen] designing {BOLTZGEN_SPEC.name} "
          f"({BOLTZGEN_NUM_DESIGNS} designs, {BOLTZGEN_PROTOCOL})\n{'='*70}", flush=True)

    row = {"model": "boltzgen", "seconds": None, "scrmsd_median": None,
           "pass_rate": None, "parse": False, "gate": False, "error": None}
    t0 = time.monotonic()
    try:
        bg._run_design(BOLTZGEN_SPEC, out, BOLTZGEN_NUM_DESIGNS, BOLTZGEN_PROTOCOL,
                       devices=None, budget=BOLTZGEN_NUM_DESIGNS, reuse=False)
    except SystemExit as e:
        row["error"] = str(e)
        return row
    row["seconds"] = time.monotonic() - t0

    cifs = sorted(out.rglob("*.cif"))
    try:
        _parse_gate(cifs, name="boltzgen")
        row["parse"] = True
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row

    # scRMSD self-consistency (harness reads out/aggregate_metrics_*.csv, the
    # isolated-refold column the shipping design_folding step already wrote).
    try:
        res = bg.score(out, sc_threshold=BOLTZGEN_SC_THRESHOLD)
    except SystemExit as e:
        row["error"] = f"designability scoring failed: {e}"
        return row
    row["scrmsd_median"], row["pass_rate"] = res["median"], res["pass_threshold"]
    row["gate"] = res["pass_threshold"] >= BOLTZGEN_MIN_PASS_RATE

    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def run_esmc(model: str, parity) -> dict:
    """Run the shipped ESMC embed path vs reference esm and gate on per-residue PCC."""
    print(f"\n{'='*70}\n[{model}] ESMC embedding parity vs reference esm "
          f"(fused-RoPE shipped path, PCC floor {ESMC_MIN_PCC})\n{'='*70}", flush=True)
    row = {"model": model, "seconds": None, "per_res_pcc": None,
           "pooled_pcc": None, "logits_pcc": None, "argmax": None,
           "gate": False, "error": None}
    t0 = time.monotonic()
    try:
        res = parity.run_esmc_parity(model, fast=FAST, pcc_threshold=ESMC_MIN_PCC)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    row["seconds"] = time.monotonic() - t0
    row["per_res_pcc"] = res["per_res_pcc"]
    row["pooled_pcc"] = res["pooled_pcc"]
    row["logits_pcc"] = res["logits_pcc"]
    row["argmax"] = res["argmax_agree"]
    row["gate"] = res["ok"]
    return row


def run_opendde_abag(keep: bool) -> dict:
    """Co-fold, parse, and DockQ-score the OpenDDE antibody-antigen leg. Returns a row.

    Mirrors ``run_model`` for the predict + parse half, then swaps the Kabsch RMSD/TM
    score for a DockQ score: the opendde-abag checkpoint co-folds a Fab + antigen
    complex, so the ground-truth bar is docking accuracy (global DockQ of the
    confidence-selected complex vs the experimental 1ahw complex), not single-chain
    CA-RMSD/TM. Reuses ``scripts/opendde_dockq.py`` (the reference DockQ tool) rather
    than re-deriving the DockQ call here.
    """
    data = OPENDDE_ABAG_DATA
    from tt_bio.main import predict_results_dir_name
    name = data.stem  # "1ahw_abag" -> opendde_results_1ahw_abag/
    out = REPO_ROOT / predict_results_dir_name("opendde-abag", name)
    if out.exists():
        shutil.rmtree(out)  # never score a stale run if this predict crashes

    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(data),
        "--model", "opendde-abag",
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--seed", str(SEED),
        *_msa_args("opendde-abag"),
        "--out_dir", str(REPO_ROOT),
    ]
    print(f"\n{'='*70}\n[opendde-abag] docking {data.name} "
          f"({SAMPLING_STEPS} steps, {DIFFUSION_SAMPLES} samples)\n{'='*70}", flush=True)

    row = {"model": "opendde-abag", "seconds": None, "dockq": None,
           "fnat": None, "parse": False, "gate": False, "error": None}
    t0 = time.monotonic()
    rc, timed_out = _run_fold(cmd, FOLD_TIMEOUT_S, cwd=REPO_ROOT)
    row["seconds"] = time.monotonic() - t0
    if timed_out:
        row["error"] = (f"predict timed out after {FOLD_TIMEOUT_S}s (flaky MSA server? set "
                        f"RELEASE_GATE_MSA_DIR to a dir with the cached a3m — see RELEASING.md)")
        return row
    if rc != 0:
        row["error"] = f"predict exited {rc}"
        return row

    struct_dir = out / "structures"
    cifs = sorted(struct_dir.glob(f"{name}*.cif")) if struct_dir.exists() else []
    try:
        _parse_gate(cifs, name=name)
        row["parse"] = True
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row

    # DockQ of the confidence-selected complex (model 0 == {name}.cif, the structure a
    # user receives) vs the experimental 1ahw complex. Shells out to the reference DockQ
    # tool under OPENDDE_DOCKQ_PYTHON (which has DockQ installed; the gate venv does not).
    conf_cif = struct_dir / f"{name}.cif"
    dockq_script = REPO_ROOT / "scripts" / "opendde_dockq.py"
    out_json = out / "dockq.json"
    try:
        dproc = subprocess.run(
            [OPENDDE_DOCKQ_PYTHON, str(dockq_script), str(conf_cif),
             str(OPENDDE_ABAG_NATIVE), "--out", str(out_json)],
            cwd=REPO_ROOT, capture_output=True, text=True)
        if dproc.returncode != 0:
            row["error"] = (f"DockQ exited {dproc.returncode}: "
                            f"{(dproc.stderr or dproc.stdout).strip()[:200]}")
            return row
        import json
        with open(out_json) as fp:
            dq = json.load(fp)
        row["dockq"] = float(dq["global_dockq"])
        # mean fnat over native interfaces, for visibility (the paratope-epitope face)
        fnats = [v.get("fnat") for v in dq["interfaces"].values()
                 if v.get("fnat") is not None]
        row["fnat"] = (sum(fnats) / len(fnats)) if fnats else None
    except Exception as e:
        row["error"] = f"DockQ eval failed: {e}"
        return row

    row["gate"] = row["dockq"] >= OPENDDE_ABAG_MIN_DOCKQ

    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def run_capacity(keep: bool, leg) -> dict:
    """Fold one capacity leg (large target, campaign-scale sample count) and check the
    peak device DRAM against its budget. Also checks the per-sample output contract,
    since a fold that quietly produces fewer structures than it was asked for is the
    other way this class of failure shows up. Returns a row."""
    from tt_bio.main import predict_results_dir_name
    yaml_rel, model, _tokens, samples, mps, budget = leg
    data = REPO_ROOT / yaml_rel
    name = data.stem
    out = REPO_ROOT / predict_results_dir_name(model, name)
    if out.exists():
        shutil.rmtree(out)

    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(data),
        "--model", model,
        "--sampling_steps", str(CAPACITY_STEPS),
        "--diffusion_samples", str(samples),
        "--max_parallel_samples", str(mps),
        "--seed", str(SEED),
        "--single_sequence",
        "--write_pae",          # the per-sample PAE files this leg counts are opt-in
        "--out_dir", str(REPO_ROOT),
    ]
    print(f"\n{'='*70}\n[capacity] {data.name} on {model}, "
          f"{samples} samples / {mps} per batch, "
          f"budget {budget:.1f} GiB\n{'='*70}", flush=True)

    row = {"model": f"capacity:{name}", "seconds": None, "peak_gib": None, "cifs": None,
           "paes": None, "gate": False, "error": None}
    log = out.parent / f"{name}_capacity.log"
    # tt_bio.tenstorrent.dram_peak appends its samples to this file. It has to be a file:
    # predict folds in a spawned worker whose stdout the live-progress view owns, so a
    # printed measurement is lost exactly when the gate collects it non-interactively.
    dram_log = out.parent / f"{name}_capacity_dram.log"
    out.parent.mkdir(parents=True, exist_ok=True)
    dram_log.unlink(missing_ok=True)
    t0 = time.monotonic()
    with open(log, "w") as fp:
        rc, timed_out = _run_fold(cmd, FOLD_TIMEOUT_S, cwd=REPO_ROOT, stdout=fp,
                                  stderr=subprocess.STDOUT,
                                  env={**os.environ, "TT_BIO_DRAM_PEAK": str(dram_log)})
    row["seconds"] = time.monotonic() - t0
    text = log.read_text(errors="replace")
    if timed_out:
        row["error"] = f"predict timed out after {FOLD_TIMEOUT_S}s"
        return row
    if rc != 0:
        tail = " / ".join(text.strip().splitlines()[-3:])[:200]
        row["error"] = f"predict exited {rc}: {tail}"
        return row

    # dram_peak writes "[DRAM] <tag>: <x> GiB used (of <y> GiB)" per new high-water mark
    dram_text = dram_log.read_text(errors="replace") if dram_log.exists() else ""
    peaks = re.findall(r"^\[DRAM\] .*?: ([0-9.]+) GiB used", dram_text, re.M)
    if not peaks:
        row["error"] = (f"no [DRAM] samples in {dram_log.name} — the dram_peak probe did not "
                        f"run, so capacity was NOT measured (an unmeasured leg is a FAIL, "
                        f"never a pass by absence of evidence)")
        return row
    # A LOWER BOUND on the true peak, not the allocator high-water mark: dram_peak() only
    # samples where model code calls it, so this is the max over instrumented points rather
    # than over time. Adding a probe call in a hot region can raise the reported number
    # without anything using more memory -- which is exactly what happened in 2026-08 when
    # main carried no tag on the eager 4-D pair transition and reported 5.90 GiB for a leg
    # whose real peak was ~11.07. Reading a true peak is not affordable (ttnn.get_memory_view
    # drains the pipeline; a 117-aa fold goes 12.0 s -> 44.7 s under a dense census), so the
    # answer is to keep the probes where the memory actually is, and to calibrate the budgets
    # below against a measurement taken with those probes present.
    row["peak_gib"] = max(float(p) for p in peaks)

    struct_dir = out / "structures"
    row["cifs"] = len(sorted(struct_dir.glob(f"{name}*.cif"))) if struct_dir.exists() else 0
    row["paes"] = len(sorted(out.rglob("*_pae.npz")))
    # One CIF per sample plus the confidence-selected copy ({name}.cif), and exactly one PAE:
    # --write_pae on main writes the best sample's pae+pde only. Per-sample PAE files
    # ({name}_model_{k}_pae.npz) are NOT a main feature -- that write path lives on the
    # AbAg-XM campaign branch, so do not gate main on it.
    if row["cifs"] != samples or row["paes"] < 1:
        row["error"] = (f"expected {samples} CIFs and >=1 PAE, "
                        f"got {row['cifs']} and {row['paes']}")
        return row
    if row["peak_gib"] > budget:
        row["error"] = f"peak {row['peak_gib']:.2f} GiB over the {budget:.1f} GiB budget"
        return row

    row["gate"] = True
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
        # The two logs sit beside the results dir (predict owns everything inside it), so
        # clear them here too rather than leaving them in the repo root after every release.
        # --keep holds on to all three for debugging a failing leg.
        log.unlink(missing_ok=True)
        dram_log.unlink(missing_ok=True)
    return row


def run_capacity_all(keep: bool) -> dict:
    """Every capacity leg as one aggregate row, for full_parity_gate's single capacity
    entry: gate is the AND, peak the worst leg, per-leg rows under 'legs'."""
    rows = [run_capacity(keep, leg) for leg in CAPACITY_LEGS]
    return {"model": "capacity",
            "seconds": sum(r["seconds"] or 0 for r in rows),
            "peak_gib": max((r["peak_gib"] or 0) for r in rows),
            "cifs": sum(r["cifs"] or 0 for r in rows),
            "paes": sum(r["paes"] or 0 for r in rows),
            "gate": all(r["gate"] for r in rows),
            "error": next((r["error"] for r in rows if r["error"]), None),
            "legs": rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    choices=list(MODELS) + ["boltzgen", "opendde-abag", "capacity"]
                    + ESMC_DEFAULT + ESMC_OPT_IN,
                    action="append",
                    help="Gate only this model (repeatable). Default: the five fold "
                         "models + boltzgen + opendde-abag + ESMC 300m/600m embed parity. "
                         "esmc-6b is opt-in (slow ~13 GB load).")
    ap.add_argument("--keep", action="store_true", help="Keep run output dirs for inspection.")
    ap.add_argument("--fast", action="store_true",
                    help="Fold with --fast so the gate exercises the block-fp8 trunk path "
                         "(bf8 weights + bf8 matmul output). Defaults off (full precision).")
    ap.add_argument("--diffusion_trace", action="store_true",
                    help="Fold boltz2 with per-step DiT ttnn trace replay on (lossless). "
                         "boltz2 only; other fold models ignore it. Defaults off.")
    args = ap.parse_args()
    global FAST, DIFFUSION_TRACE
    FAST = args.fast
    DIFFUSION_TRACE = args.diffusion_trace

    # A lone P300 Blackhole chip is a custom topology: ttnn refuses to open
    # it without a 1x1 mesh-graph descriptor. The predict/embed CLIs set this
    # per worker / in-process, but the gen subprocess and this process's
    # in-process ESMC embed leg (esmc.embed_sequences, bypassing the embed
    # CLI) do not -- set it once here so every leg inherits it. Mirrors
    # scripts/perf_regression.py and tt_bio/main.py's embed command.
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    models = args.model or list(MODELS) + ["boltzgen", "opendde-abag", "capacity"] + ESMC_DEFAULT
    fold_models = [m for m in models if m in MODELS]
    want_boltzgen = "boltzgen" in models
    want_opendde_abag = "opendde-abag" in models
    want_capacity = "capacity" in models
    esmc_models = [m for m in models if m in ESMC_DEFAULT + ESMC_OPT_IN]

    rows = []
    if fold_models:
        if not DATA.exists():
            sys.exit(f"missing gate target {DATA}")
        if not GROUND_TRUTH.exists():
            sys.exit(f"missing ground truth {GROUND_TRUTH}")
        harness = _load_structure_harness()
        rows = [run_model(m, harness, args.keep) for m in fold_models]

    all_pass = True
    if rows:
        print(f"\n{'#'*78}\nRELEASE GATE — {DATA.name} ({NAME}), "
              f"{SAMPLING_STEPS} steps / {DIFFUSION_SAMPLES} samples, seed {SEED}\n{'#'*78}")
        print(f"{'model':<15}{'RMSD (A)':>10}{'TM':>8}{'floor':>16}{'wall':>9}  result")
        for r in rows:
            th = MODELS[r["model"]]
            floor = f"<={th['max_rmsd']}/>={th['min_tm']}"
            rmsd = f"{r['rmsd']:.3f}" if r["rmsd"] is not None else "  -  "
            tm = f"{r['tm']:.3f}" if r["tm"] is not None else "  -  "
            wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
            verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r["error"] else "FAIL"
            all_pass &= r["gate"]
            print(f"{r['model']:<15}{rmsd:>10}{tm:>8}{floor:>16}{wall:>9}  {verdict}")
        print(f"{'#'*78}")
        print("GATE PASS — all models cleared parse + ground-truth floor" if all_pass
              else "GATE FAIL — a model missed parse or the ground-truth floor (see above)")

    if want_boltzgen:
        bg = _load_designability_harness()
        br = run_boltzgen(bg, args.keep)
        print(f"\n{'#'*78}\nRELEASE GATE — {BOLTZGEN_SPEC.name} (boltzgen), "
              f"{BOLTZGEN_NUM_DESIGNS} designs, {BOLTZGEN_PROTOCOL}\n{'#'*78}")
        print(f"{'model':<15}{'scRMSD (A)':>12}{'pass rate':>12}{'floor':>18}{'wall':>9}  result")
        floor = f"<={BOLTZGEN_SC_THRESHOLD}A>={BOLTZGEN_MIN_PASS_RATE*100:.0f}%"
        scrmsd = f"{br['scrmsd_median']:.3f}" if br["scrmsd_median"] is not None else "  -  "
        pr = f"{br['pass_rate']*100:.0f}%" if br["pass_rate"] is not None else "  -  "
        wall = f"{br['seconds']:.0f}s" if br["seconds"] is not None else "-"
        verdict = "PASS" if br["gate"] else f"FAIL ({br['error']})" if br["error"] else "FAIL"
        all_pass &= br["gate"]
        print(f"{br['model']:<15}{scrmsd:>12}{pr:>12}{floor:>18}{wall:>9}  {verdict}")
        print(f"{'#'*78}")
        print("GATE PASS — boltzgen designs cleared parse + designability floor" if br["gate"]
              else "GATE FAIL — boltzgen missed parse or the designability floor (see above)")

    if want_opendde_abag:
        if not OPENDDE_ABAG_DATA.exists():
            sys.exit(f"missing opendde-abag gate target {OPENDDE_ABAG_DATA}")
        if not OPENDDE_ABAG_NATIVE.exists():
            sys.exit(f"missing opendde-abag ground truth {OPENDDE_ABAG_NATIVE}")
        ar = run_opendde_abag(args.keep)
        print(f"\n{'#'*78}\nRELEASE GATE — {OPENDDE_ABAG_DATA.name} (opendde-abag), "
              f"{SAMPLING_STEPS} steps / {DIFFUSION_SAMPLES} samples, seed {SEED}\n{'#'*78}")
        print(f"{'model':<15}{'global DockQ':>14}{'mean fnat':>11}{'floor':>10}{'wall':>9}  result")
        floor = f">={OPENDDE_ABAG_MIN_DOCKQ}"
        dq = f"{ar['dockq']:.3f}" if ar["dockq"] is not None else "  -  "
        fn = f"{ar['fnat']:.3f}" if ar["fnat"] is not None else "  -  "
        wall = f"{ar['seconds']:.0f}s" if ar["seconds"] is not None else "-"
        verdict = "PASS" if ar["gate"] else f"FAIL ({ar['error']})" if ar["error"] else "FAIL"
        all_pass &= ar["gate"]
        print(f"{ar['model']:<15}{dq:>14}{fn:>11}{floor:>10}{wall:>9}  {verdict}")
        print(f"{'#'*78}")
        print("GATE PASS — opendde-abag cleared parse + DockQ floor" if ar["gate"]
              else "GATE FAIL — opendde-abag missed parse or the DockQ floor (see above)")

    if want_capacity:
        for leg in CAPACITY_LEGS:
            if not (REPO_ROOT / leg[0]).exists():
                sys.exit(f"missing capacity gate target {leg[0]}")
        rows = [run_capacity(args.keep, leg) for leg in CAPACITY_LEGS]
        print(f"\n{'#'*78}\nRELEASE GATE — capacity "
              f"({', '.join(f'{REPO_ROOT.joinpath(l[0]).stem} on {l[1]}' for l in CAPACITY_LEGS)}, "
              f"{CAPACITY_STEPS} steps)\n{'#'*78}")
        print(f"{'leg':<22}{'peak DRAM':>11}{'CIFs':>7}{'PAEs':>7}{'budget':>11}{'wall':>9}  result")
        for leg, cr in zip(CAPACITY_LEGS, rows):
            pk = f"{cr['peak_gib']:.2f} GiB" if cr["peak_gib"] is not None else "  -  "
            cf = str(cr["cifs"]) if cr["cifs"] is not None else "-"
            pa = str(cr["paes"]) if cr["paes"] is not None else "-"
            wall = f"{cr['seconds']:.0f}s" if cr["seconds"] is not None else "-"
            verdict = "PASS" if cr["gate"] else f"FAIL ({cr['error']})" if cr["error"] else "FAIL"
            all_pass &= cr["gate"]
            print(f"{cr['model']:<22}{pk:>11}{cf:>7}{pa:>7}{f'<={leg[5]:.1f} GiB':>11}"
                  f"{wall:>9}  {verdict}")
        print(f"{'#'*78}")
        print("GATE PASS — largest-input folds fit the DRAM budget and wrote every sample"
              if all(cr["gate"] for cr in rows) else
              "GATE FAIL — capacity regression at the largest supported input (see above)")

    if esmc_models:
        if "ESM_ROOT" not in os.environ:
            sys.exit("ESMC parity leg needs ESM_ROOT (path to the esm clone for tests/esmc_reference.py)")
        parity = _load_esmc_parity_harness()
        erows = [run_esmc(m, parity) for m in esmc_models]
        esmc_pass = all(r["gate"] for r in erows)
        print(f"\n{'#'*78}\nRELEASE GATE — ESMC embedding parity (fused-RoPE shipped path), "
              f"PCC floor {ESMC_MIN_PCC}\n{'#'*78}")
        print(f"{'model':<12}{'per-res PCC':>13}{'pooled':>9}{'logits':>9}{'argmax':>9}{'wall':>9}  result")
        for r in erows:
            pr = f"{r['per_res_pcc']:.5f}" if r["per_res_pcc"] is not None else "  -  "
            po = f"{r['pooled_pcc']:.5f}" if r["pooled_pcc"] is not None else "  -  "
            lo = f"{r['logits_pcc']:.5f}" if r["logits_pcc"] is not None else "  -  "
            am = f"{r['argmax']:.4f}" if r["argmax"] is not None else "  -  "
            wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
            verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r["error"] else "FAIL"
            all_pass &= r["gate"]
            print(f"{r['model']:<12}{pr:>13}{po:>9}{lo:>9}{am:>9}{wall:>9}  {verdict}")
        print(f"{'#'*78}")
        print("GATE PASS — ESMC embed path cleared the per-residue PCC floor" if esmc_pass
              else "GATE FAIL — an ESMC model missed the per-residue PCC floor (see above)")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
