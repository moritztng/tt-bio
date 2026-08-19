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

The l1-budget leg is the only one that gates a PART, not a number. Every L1-edge budget
in ``tt_bio/tenstorrent.py`` was fitted on a 130-core p150a, and ``_apply_grid_thresholds``
returns early on any grid of 110 cores or more, so a 110-core Blackhole (p300/p300c) runs
budgets fitted for 130 cores and the trimul in-projection's circular buffers stop fitting
beside the pair tensors (issue #11, Taylor Singletary). The legs above cannot see that: they
compare numbers, and a part that dies at program creation produces no numbers. This leg runs
the budget arithmetic for every part class we have a measured per-core-L1 figure for, and
folds the input that died on his p300c across this part's grid ladder. THE RULE it enforces:
a part-specific resource figure entering ``tenstorrent.py`` gets a row in ``L1_BUDGET_PARTS``
in the same commit.

This is the *accuracy* leg of the release gate. The *UX* leg lives in
``scripts/ux_regression.py`` (live-progress phases, output parsing, CLI shape) —
see RELEASING.md. The two are independent; both must exit 0 before a tag.
"""

import argparse
import hashlib
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
# Budgets, in GiB of device DRAM. Leg 1's measured peak on a Blackhole p150a is 8.73 GiB
# (tt-bio-large-target-oom-rootcause census-tagged the true high-water mark; the prior 5.90
# GiB reading was an under-read at the same probe point on untagged code, and instrumented
# pre-fix main reads 11.07 GiB there -- this branch is the lower-footprint version). 10.5
# leaves the same ~19% headroom as before: loose enough that it does not chase run-to-run
# noise (the shapes are deterministic, so there is very little), tight enough to catch the
# small regressions too -- re-replicating just the atom-transformer pair bias and the
# windowed atom pair tensor would add ~0.9 GiB, and re-replicating the DiT pair biases would
# add ~9.6 GiB. Env-tunable so a card with a different budget can gate against its own.
#
# Leg 2 is the structural-token ceiling: opendde-abag's refiner runs the pair track at
# ~1.9x the residue count, which is where the WH 12 GiB OOM exclusions came from. Fewer
# samples than leg 1 because the pair-track peak this leg guards is sample-count
# independent; the sample-scaled diffusion footprint is leg 1's job.
CAPACITY_LEGS = [
    # (yaml, model, residue tokens, samples, mps, budget GiB)
    ("examples/abag_pilot_expansion/9j4c_abag.yaml", "protenix-v2", 1095, 50, 5,
     float(os.environ.get("RELEASE_GATE_CAPACITY_MAX_GIB", "10.5"))),
    ("examples/abag_xm/9ivj.yaml", "opendde-abag", 891, 8, 2,
     float(os.environ.get("RELEASE_GATE_CAPACITY_MAX_GIB_2", "12.0"))),
]


# ---------------------------------------------------------------------------
# L1-budget leg — the structural blind spot issue #11 fell through.
#
# Every L1-edge budget in tt_bio/tenstorrent.py was fitted on a 130-core p150a, and
# `_apply_grid_thresholds` returns early on any grid of >= 110 cores ("keep Blackhole
# baseline values"). A 110-core Blackhole (p300/p300c) therefore runs budgets fitted for
# 130 cores: the same bytes on fewer cores, so the trimul in-projection's static circular
# buffers take more per core and the width the budget picks does not fit beside the pair
# tensors live at the call site. Nothing in the other legs can see that. They compare
# numbers, and a part that dies at program creation produces no numbers to compare.
#
# One row per part class we have a per-core unreserved-L1 figure for, with the compute grid
# tt-bio selects on it. THE RULE, and the reason this leg exists: a part-specific resource
# figure entering tenstorrent.py gets a row here in the same commit.
L1_BUDGET_PARTS = (
    # (name, grid, per-core unreserved L1 bytes, provenance)
    ("p150a", (13, 10), 1532416,
     "measured on pc 2026-08-19, ttnn.get_max_worker_l1_unreserved_size()"),
    ("p300c", (11, 10), 1532416,
     "measured on tt-quietbox2 device 0 2026-08-19, same call. Identical to p150a — the "
     "p300 difference that broke issue #11 is core count (110 vs 130), not per-core L1"),
    ("wormhole", (8, 8), 1466080, "_WH_MEASURED_L1_PER_CORE (the L1 the WH re-fit was measured at)"),
    ("wormhole-full", (8, 8), 1572864, "_WH_FULL_L1_PER_CORE (1.5 MiB, the WH scaling reference)"),
)

# Trimul chunk widths MEASURED to clash on real hardware, per part and call shape
# (seq, hidden, batch). Not derived: each is a tt-metal "clash with L1 buffers" throw
# observed in a log. The arithmetic leg asserts the code can get below every one of them.
L1_BUDGET_MEASURED_CLASHES = (
    ("p300c", (140, 256, 1), 256,
     "issue #11 and tt-quietbox2 native 11x10 2026-08-19: 'Statically allocated circular "
     "buffers in program N clash with L1 buffers', L1 buffer at 1155072, CB region ends "
     "1159680 — byte-identical addresses on Taylor Singletary's p300c and on ours"),
)

# Shapes the arithmetic leg sweeps. The invariant it checks is shape-independent, so it
# sweeps rather than picking a favourite: every 32-multiple seq the trunk can present up
# to the L1 residency ceiling, at both trimul hidden widths.
L1_BUDGET_SEQ_LENS = tuple(range(64, 641, 32))
L1_BUDGET_HIDDEN = (128, 256)

# The live leg folds Taylor's own target: 107 aa + CCD ligand SB3, msa: empty, the input
# that died on his p300c. Cheap enough for a default arm (~15 s a grid).
L1_BUDGET_DATA = REPO_ROOT / "examples" / "affinity_fkg.yaml"
L1_BUDGET_MODEL = "protenix-v2"
DEFAULT_ARMS = ("boltzgen", "opendde-abag", "capacity", "l1-budget")

L1_BUDGET_STEPS = 200
# The width measured to fit at the issue-#11 call shape on a 110-core grid: the
# clash-and-narrow fold lands here, so a fold pinned here from the start is the
# same computation without the interrupted attempt.
L1_BUDGET_CAP = 128


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


def _preflight_msa_cache(models: list) -> None:
    """Fail before any device work if the offline MSA dir cannot serve a target it will fold.

    A missing a3m does not surface as a missing-input error. The fold falls through to
    colabfold_search, which is not installed on the gate hosts, so the leg dies with
    "colabfold_search not found" and the summary reads as a missed accuracy floor: an hour
    into the run, on an arm whose numbers were never computed. That is how a
    seeded-for-7ROA-only RELEASE_GATE_MSA_DIR failed the opendde-abag arm on the first
    v0.6.4 gate run. Keyed off MSA_DEFAULT_MODELS, the same source of truth _msa_args uses,
    so the check cannot drift from what the folds actually request.
    """
    if not MSA_DIR:
        return
    import yaml
    from tt_bio.main import MSA_DEFAULT_MODELS

    targets = []
    if any(m in MODELS and m in MSA_DEFAULT_MODELS for m in models):
        targets.append(DATA)
    if "opendde-abag" in models:
        targets.append(OPENDDE_ABAG_DATA)

    missing = []
    for path in targets:
        if not path.exists():
            continue
        doc = yaml.safe_load(path.read_text()) or {}
        for entry in doc.get("sequences") or []:
            prot = entry.get("protein") if isinstance(entry, dict) else None
            seq = (prot or {}).get("sequence")
            if not seq:
                continue
            a3m = Path(MSA_DIR) / f"{hashlib.sha256(seq.encode()).hexdigest()[:16]}.a3m"
            if not (a3m.exists() and a3m.stat().st_size > 0):
                cid = (prot or {}).get("id", "?")
                missing.append(f"{path.name} chain {cid} ({len(seq)} aa) -> {a3m}")
    if missing:
        sys.exit(
            f"RELEASE_GATE_MSA_DIR={MSA_DIR} cannot serve every MSA-dependent gate target.\n"
            "Missing cached a3m (the file name is sha256(sequence)[:16]):\n  "
            + "\n  ".join(missing)
            + "\nSeed them, or unset RELEASE_GATE_MSA_DIR to fold against the ColabFold "
              "server. See RELEASING.md."
        )


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


# --- L1-budget leg (issue #11) ------------------------------------------------

# Module globals `tt_bio.tenstorrent._apply_grid_thresholds` rewrites, plus the active grid
# and the two clash memos. The arithmetic leg installs a part's grid/L1, so it must put the
# module back exactly as it found it: the live leg and every other in-process leg share it.
_L1_BUDGET_SAVED = (
    "_IS_SMALL_GRID", "SEQ_LEN_MORE_CHUNKING", "TRANSITION_BATCH_CHUNKING_THRESHOLD",
    "TRANSITION_W_CHUNKING_THRESHOLD", "TRIANGLE_ATT_CHUNK_SIZE_FAST", "TRANSITION_W_CHUNK_SIZE",
    "TRIANGLE_MULT_L1_MAX_SEQ_FAST", "SMALL_GRID_SEQ_TILE", "SMALL_GRID_PAIR_TILE_AREA",
    "SMALL_GRID_MSA_TILE_AREA", "TRIANGLE_MULT_L1_MAX_SEQ", "TRANSITION_L1_CHUNK_BYTES_PER_CORE",
    "TRIATT_CHUNK_L1_SPILL_BYTES", "COMPUTE_GRID_MAIN", "CORE_GRID_MAIN",
)


def _l1_budget_ladder(tt, seq: int, hidden: int, batch: int, part: str) -> list:
    """Walk the trimul retry ladder for one call shape. Returns failure strings.

    The invariant: a width that clashes on this part must be escapable. The budget picks a
    width; if that width throws at program creation the next pick must be strictly narrower,
    still a divisor of `hidden` (narrowing is only bit-exact because the width partitions an
    independent-channel sum), and the ladder must reach the minimum width in finitely many
    steps, where the shape leaves L1 for DRAM. Issue #11 was exactly the absence of this:
    the 130-core budget picked 256 on a 110-core grid, 256 threw, and nothing in the code
    could pick anything else, so the fold was dead.
    """
    import ttnn
    fails = []
    shape = f"seq={seq} hidden={hidden} batch={batch}"
    tt._TRIMUL_CHUNK_CLASH.clear()
    tt._TRIMUL_DRAM_SHAPES.clear()
    width = tt._trimul_chunk_size(seq, hidden, batch)
    if hidden % width:
        fails.append(f"{part} {shape}: budget picked width {width}, not a divisor of hidden "
                     f"{hidden} — narrowing would not be bit-exact")
    steps = 0
    while width > tt.TRIANGLE_MULT_CHUNK_SIZE:
        steps += 1
        if steps > 16:
            fails.append(f"{part} {shape}: retry ladder did not reach the minimum width in "
                         f"16 steps (stuck at {width})")
            break
        tt._record_trimul_clash(seq, hidden, batch, width)
        nxt = tt._trimul_chunk_size(seq, hidden, batch)
        if nxt >= width:
            fails.append(f"{part} {shape}: width {width} clashed and the budget still picks "
                         f"{nxt} — a clash on this part has no way out (issue #11)")
            break
        if hidden % nxt:
            fails.append(f"{part} {shape}: narrowed {width} -> {nxt}, not a divisor of hidden "
                         f"{hidden} — narrowing would not be bit-exact")
            break
        width = nxt
    # At the minimum width the only remaining move is to leave L1 altogether.
    tt._TRIMUL_DRAM_SHAPES.add(seq)
    if tt._triangle_mul_memory_config(seq).buffer_type != ttnn.BufferType.DRAM:
        fails.append(f"{part} {shape}: at the minimum width the shape cannot fall back to "
                     f"DRAM — the ladder has no floor")
    tt._TRIMUL_CHUNK_CLASH.clear()
    tt._TRIMUL_DRAM_SHAPES.clear()
    return fails


def run_l1_budget_static() -> dict:
    """Arithmetic leg — no device, no fold. Runs tt_bio.tenstorrent's own trimul budget at
    every part class in L1_BUDGET_PARTS and checks the issue-#11 invariant on each."""
    import ttnn
    import tt_bio.tenstorrent as tt

    row = {"model": "l1-budget:arith", "seconds": None, "parts": len(L1_BUDGET_PARTS),
           "checks": 0, "gate": False, "error": None, "fails": []}
    t0 = time.monotonic()
    # A build with no clash-narrowing mechanism at all fails the leg, it does not crash it:
    # that build is exactly the one issue #11 was filed against.
    missing = [n for n in ("_record_trimul_clash", "_TRIMUL_CHUNK_CLASH", "_TRIMUL_DRAM_SHAPES")
               if not hasattr(tt, n)]
    if missing:
        row["seconds"] = time.monotonic() - t0
        row["fails"] = [f"tt_bio.tenstorrent has no {', '.join(missing)}: a trimul chunk width "
                        f"that clashes on this part cannot be narrowed, so the fold dies at "
                        f"program creation with no way out (issue #11)"]
        return row
    saved = {n: getattr(tt, n) for n in _L1_BUDGET_SAVED}
    saved_clash = dict(tt._TRIMUL_CHUNK_CLASH)
    saved_dram = set(tt._TRIMUL_DRAM_SHAPES)
    real_l1 = ttnn.get_max_worker_l1_unreserved_size
    fails = []
    try:
        for part, grid, l1, _prov in L1_BUDGET_PARTS:
            ttnn.get_max_worker_l1_unreserved_size = lambda _l1=l1: _l1
            tt.COMPUTE_GRID_MAIN = grid
            tt.CORE_GRID_MAIN = ttnn.CoreGrid(y=grid[1], x=grid[0])
            tt._apply_grid_thresholds(grid)
            for name in ("_trimul_l1_max_seq", "_triangle_mul_program_config"):
                fn = getattr(tt, name, None)
                if hasattr(fn, "cache_clear"):
                    fn.cache_clear()
            for hidden in L1_BUDGET_HIDDEN:
                for seq in L1_BUDGET_SEQ_LENS:
                    fails += _l1_budget_ladder(tt, seq, hidden, 1, part)
                    row["checks"] += 1
            for cpart, (seq, hidden, batch), width, _src in L1_BUDGET_MEASURED_CLASHES:
                if cpart != part:
                    continue
                # A width measured to clash on this part must be escapable, from whatever
                # the budget picks down to below it.
                tt._TRIMUL_CHUNK_CLASH.clear()
                tt._record_trimul_clash(seq, hidden, batch, width)
                got = tt._trimul_chunk_size(seq, hidden, batch)
                tt._TRIMUL_CHUNK_CLASH.clear()
                row["checks"] += 1
                if got >= width:
                    fails.append(f"{part} seq={seq} hidden={hidden}: width {width} is MEASURED "
                                 f"to clash here and the budget still picks {got}")
        # The rule the leg exists to enforce: no grid tt-bio can select is unrepresented.
        table_grids = {g for _n, g, _l, _p in L1_BUDGET_PARTS}
        for grid in ((tt.COMPUTE_GRID_X_13, tt.COMPUTE_GRID_Y),
                     (tt.COMPUTE_GRID_X_11, tt.COMPUTE_GRID_Y)):
            if grid not in table_grids:
                fails.append(f"grid {grid[0]}x{grid[1]} is selectable by "
                             f"_configure_active_compute_grid but has no L1_BUDGET_PARTS row — "
                             f"add the part and its measured per-core L1 in the same commit")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
    finally:
        ttnn.get_max_worker_l1_unreserved_size = real_l1
        for n, v in saved.items():
            setattr(tt, n, v)
        tt._TRIMUL_CHUNK_CLASH.clear()
        tt._TRIMUL_CHUNK_CLASH.update(saved_clash)
        tt._TRIMUL_DRAM_SHAPES.clear()
        tt._TRIMUL_DRAM_SHAPES.update(saved_dram)
    row["seconds"] = time.monotonic() - t0
    row["fails"] = fails
    row["gate"] = not fails and row["error"] is None
    return row


def _l1_budget_physical_grid() -> tuple:
    """The running part's compute grid, read in a throwaway subprocess so this driver never
    holds a device while the fold legs below need it."""
    code = ("import ttnn;d=ttnn.open_device(device_id=0);g=d.compute_with_storage_grid_size();"
            "print(int(g.x),int(g.y));ttnn.close_device(d)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=REPO_ROOT, timeout=600)
    for line in reversed(out.stdout.strip().splitlines()):
        parts = line.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            return (int(parts[0]), int(parts[1]))
    raise RuntimeError(f"could not read the compute grid: {out.stdout[-500:]} {out.stderr[-500:]}")


def _l1_budget_fold(label: str, grid, cap: int, keep: bool) -> dict:
    """One fkg fold at a forced grid and/or a forced trimul chunk width."""
    out = REPO_ROOT / f"l1_budget_gate_{label}"
    if out.exists():
        shutil.rmtree(out)
    env = os.environ.copy()
    env.pop("TT_BIO_FORCE_GRID", None)
    if grid is not None:
        env["TT_BIO_FORCE_GRID"] = f"{grid[0]},{grid[1]}"
    if cap:
        env["TT_BIO_TRIMUL_CHUNK_CAP"] = str(cap)
    else:
        env.pop("TT_BIO_TRIMUL_CHUNK_CAP", None)
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(L1_BUDGET_DATA),
           "--model", L1_BUDGET_MODEL, "--single_sequence",
           "--sampling_steps", str(L1_BUDGET_STEPS), "--diffusion_samples", "1",
           "--seed", str(SEED), "--out_dir", str(out), "--debug"]
    print(f"\n{'='*70}\n[l1-budget] folding {L1_BUDGET_DATA.name} on {L1_BUDGET_MODEL}, "
          f"grid {'native' if grid is None else f'{grid[0]}x{grid[1]}'}"
          f"{f', trimul width capped at {cap}' if cap else ''}\n{'='*70}", flush=True)

    row = {"model": f"l1-budget:{label}", "grid": grid, "cap": cap, "seconds": None,
           "clashes": None, "md5": None, "gate": False, "error": None}
    log = REPO_ROOT / f"l1_budget_gate_{label}.log"
    t0 = time.monotonic()
    with open(log, "wb") as fh:
        rc, timed_out = _run_fold(cmd, FOLD_TIMEOUT_S, cwd=REPO_ROOT, env=env,
                                  stdout=fh, stderr=subprocess.STDOUT)
    row["seconds"] = time.monotonic() - t0
    text = log.read_text(errors="replace")
    # Every clash tt-metal threw. The fix catches these and retries narrower, so a nonzero
    # count is not a failure -- an escaped one shows up as a nonzero exit code below.
    row["clashes"] = text.count("clash with L1 buffers")
    if timed_out:
        row["error"] = f"predict timed out after {FOLD_TIMEOUT_S}s"
        return row
    if rc != 0:
        tail = [l for l in text.splitlines() if "clash with L1 buffers" in l]
        row["error"] = (f"predict exited {rc}"
                        + (f" — an L1/CB clash escaped: {tail[-1][-160:]}" if tail else ""))
        return row
    results = out / _l1_budget_results_dir()
    cifs = sorted((results / "structures").glob("*.cif")) if (results / "structures").exists() else []
    if not cifs:
        row["error"] = f"predict wrote no CIF under {results}"
        return row
    try:
        _parse_gate(cifs, name=L1_BUDGET_DATA.stem)
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row
    h = hashlib.md5()
    for cif in cifs:
        h.update(cif.read_bytes())
    row["md5"] = h.hexdigest()
    row["gate"] = True
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
        log.unlink(missing_ok=True)
    return row


def _l1_budget_results_dir() -> str:
    from tt_bio.main import predict_results_dir_name
    return predict_results_dir_name(L1_BUDGET_MODEL, L1_BUDGET_DATA.stem)


def run_l1_budget_fold(keep: bool) -> list:
    """Live leg — fold Taylor's target on this part's grid ladder, and prove the retry is
    numerics-neutral.

    Two things no other leg can see:
      1. Grid classes. Every grid in L1_BUDGET_PARTS that fits inside the physical grid gets
         a fold, so the >=110-core baseline-budget path and the <110-core scaled path are both
         exercised on whatever part the gate runs on. On a 110-core p300 the native leg IS
         issue #11's crash: pre-fix it dies here.
      2. Retry neutrality. The cold native fold reaches its width by clashing and narrowing;
         the capped fold starts at that narrow width and never clashes. Same numbers or the
         retry is contaminated by the interrupted attempt.
    """
    phys = _l1_budget_physical_grid()
    legs = [("native", None, 0)]
    for name, grid, _l1, _prov in L1_BUDGET_PARTS:
        if grid == phys or grid[0] > phys[0] or grid[1] > phys[1]:
            continue  # native already covers phys; a bigger grid than the part has cannot be forced
        if any(grid == g for _lbl, g, _c in legs):
            continue
        legs.append((f"{grid[0]}x{grid[1]}", grid, 0))
    legs.append(("narrow", None, L1_BUDGET_CAP))
    rows = [_l1_budget_fold(label, grid, cap, keep) for label, grid, cap in legs]

    # Neutrality: the clash-and-narrow fold must match the never-clashed one byte for byte.
    cold = next((r for r in rows if r["model"] == "l1-budget:native"), None)
    narrow = next((r for r in rows if r["model"] == "l1-budget:narrow"), None)
    if cold and narrow and cold["md5"] and narrow["md5"] and cold["md5"] != narrow["md5"]:
        narrow["gate"] = False
        narrow["error"] = (f"retry is NOT numerics-neutral: clash-and-narrow {cold['md5']} != "
                           f"width-capped {narrow['md5']}")
    return rows


def main() -> int:
    # Scorers, folds and predict CLIs we spawn arm their parent-death guard off this,
    # so none of them can outlive this driver still holding a card. Inherited through
    # every spawn path in this file (see tt_bio/device_lease.py:arm_orphan_guard).
    os.environ["TT_BIO_PARENT_PID"] = str(os.getpid())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    choices=list(MODELS) + list(DEFAULT_ARMS)
                    + ESMC_DEFAULT + ESMC_OPT_IN,
                    action="append",
                    help="Gate only this model (repeatable). Default: the five fold "
                         "models + boltzgen + opendde-abag + capacity + l1-budget + "
                         "ESMC 300m/600m embed parity. "
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

    models = args.model or list(MODELS) + list(DEFAULT_ARMS) + ESMC_DEFAULT
    fold_models = [m for m in models if m in MODELS]
    want_boltzgen = "boltzgen" in models
    want_opendde_abag = "opendde-abag" in models
    want_capacity = "capacity" in models
    want_l1_budget = "l1-budget" in models
    esmc_models = [m for m in models if m in ESMC_DEFAULT + ESMC_OPT_IN]
    _preflight_msa_cache(models)

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

    if want_l1_budget:
        if not L1_BUDGET_DATA.exists():
            sys.exit(f"missing l1-budget gate target {L1_BUDGET_DATA}")
        ar = run_l1_budget_static()
        frows = run_l1_budget_fold(args.keep)
        md5s = {r["md5"] for r in frows if r["md5"]}
        if len(md5s) > 1:
            for r in frows:
                r["gate"] = False
                r["error"] = r["error"] or (
                    f"grid/width ladder is NOT numerics-neutral: {sorted(md5s)}")
        print(f"\n{'#'*78}\nRELEASE GATE — L1 budgets vs part "
              f"({L1_BUDGET_DATA.name} on {L1_BUDGET_MODEL}, "
              f"{len(L1_BUDGET_PARTS)} part classes)\n{'#'*78}")
        print(f"{'leg':<22}{'clashes':>9}{'CIF md5':>36}{'wall':>9}  result")
        arv = "PASS" if ar["gate"] else "FAIL"
        checks = f"{ar['checks']} checks / {ar['parts']} parts"
        print(f"{'l1-budget:arith':<22}{'-':>9}{checks:>36}{ar['seconds']:>8.0f}s  {arv}")
        for f in ar["fails"][:8]:
            print(f"    {f}")
        if ar["error"]:
            print(f"    {ar['error']}")
        for r in frows:
            cl = str(r["clashes"]) if r["clashes"] is not None else "-"
            md5 = r["md5"] or "  -  "
            wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
            verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r["error"] else "FAIL"
            print(f"{r['model']:<22}{cl:>9}{md5:>36}{wall:>9}  {verdict}")
        l1_pass = ar["gate"] and all(r["gate"] for r in frows)
        all_pass &= l1_pass
        print(f"{'#'*78}")
        print("GATE PASS — every part class can narrow out of a clash, and the ladder is "
              "numerics-neutral" if l1_pass else
              "GATE FAIL — a part class cannot escape an L1/CB clash, or the ladder changed "
              "the numbers (see above)")

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
