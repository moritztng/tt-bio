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

Size-ladder is the size-generality arm. STANDING RULE: a perf lever may not land
default-ON on the strength of ONE sequence length, and any threshold constant in
``tt_bio/tenstorrent.py`` carries a validity range stated where it is defined.
Between 2026-08-13 and 2026-08-19 every perf decision on main was screened at
512 aa only, and the 512-tuned L1 gates went silently dark above 640 aa — no
error, no log line, the fold just got slower (N^2.0 -> N^3.6 over 512->768). A
one-off sweep does not survive contact with merges, so the check is a standing
arm: fold each model at 256/512/768 aa through ``scripts/lever_census.py`` and
fail when the fired/dark lever set or the log-log runtime exponent between
rungs drifts from the checked-in baseline (``docs/size_ladder_baseline.json``).
It is a change detector, not a purity check: legitimately dark levers carry a
one-line exemption reason in the baseline, and re-recording
(``--size-ladder-record``) is an explicit human action that runs all three
rungs, so flipping a default forces measuring three sizes. Baselines are per
card type, same discipline as ``docs/perf_baselines.json``. See
docs/size-generality.md.

    # gate everything (five fold models + BoltzGen designability + ESMC embed parity
    # + OpenDDE-abag docking + capacity + size-ladder) on card 1
    TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> ESM_ROOT=/path/to/esm \
        OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python \
        python scripts/release_gate.py
    # one leg
    python scripts/release_gate.py --model protenix-v2
    python scripts/release_gate.py --model boltzgen
    python scripts/release_gate.py --model esmc-300m
    python scripts/release_gate.py --model opendde-abag
    python scripts/release_gate.py --model capacity
    python scripts/release_gate.py --model size-ladder
    # re-record the size-ladder baseline after an intentional size-affecting change
    python scripts/release_gate.py --model size-ladder --size-ladder-record

Exit code 0 iff every requested model PASSES its gate; 1 otherwise. Runs on the
device serially (one card context per run); no CPU shortcut for the fold/design.

This is the *accuracy* leg of the release gate. The *UX* leg lives in
``scripts/ux_regression.py`` (live-progress phases, output parsing, CLI shape) —
see RELEASING.md. The two are independent; both must exit 0 before a tag.
"""

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import socket
import statistics
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

# --- size-ladder leg ---------------------------------------------------------
# The accuracy legs above compare NUMBERS at one small target, so a perf lever
# that only fires at the sequence length it was tuned at — and silently stops
# firing at every other length — is invisible to them. That is not hypothetical:
# every perf decision of 2026-08-13..19 was screened at 512 aa only, and the
# 512-tuned L1 gates (K2 fill_preconditions, _TRANSPOSE_L1_HEADROOM, the SDPA
# q-chunk budget) went dark outside that window with no error and no log line.
# This leg folds each model at three rungs through scripts/lever_census.py and
# fails when the fired/dark lever set or the runtime scaling exponent drifts
# from the checked-in baseline. It is a CHANGE DETECTOR against a recorded
# baseline, not a purity check: some levers are legitimately dark at some sizes,
# so the baseline ships today's dark set with a one-line exemption reason per
# dark lever, and re-recording (--size-ladder-record) is an explicit human
# action that runs all three rungs — flipping a default forces measuring three
# sizes. The standing rule this enforces is in the module docstring.
#
# Rungs 256/512/768: 512 is the anchor every lever was tuned at; 256 is inside
# the compute-bound regime (128 aa is fixed-cost dominated — ~3 s of prep /
# confidence / save against ~5 s of fold — and sits below grid saturation, so
# half its dark levers are uninteresting); 768 is where the measured N^3.6
# cliff and the SDPA q-chunk overflow live. 1024 is excluded: OpenFold3 OOMs
# there on allocation COUNT, so the arm would be red on arrival for one model,
# which is how arms get disabled. Override for a single-rung debug run:
# RELEASE_GATE_SIZE_RUNGS=640.
#
# 640 is the fourth rung and it is not decoration. 256/512/768/1024 all have a
# padded length that 256 divides, and _capped_sdpa_chunk_size returns 256, so
# they all sit on the lattice the fused K1/K2 kernel is SERVED on. 6dbdcf1f
# records the same kernel silently declining at padded 448, 576, 640, 704, 832,
# 896 and 960 while 256/512/768/1024 were served: a ladder built only from
# multiples of 256 holds "N padded is a multiple of the SDPA chunk" constant at
# every rung and is blind to that whole defect class (the size-axis form of
# correctness-sweep-tiled-fixture-measures-one-input). 640 is the off-lattice
# control, and it is the rung the arm's own RED proof fires at.
#
# Fold config: --single_sequence --sampling_steps 6 --diffusion_samples 1
# --seed 0. Single-sequence makes the arm hermetic (no MSA server, no
# RELEASE_GATE_MSA_DIR precondition, nothing that can fail for a reason
# unrelated to what is gated). None of the census levers live in an MSA module,
# so this costs no lever coverage; the price is that a cliff living ONLY in the
# MSA module is invisible here. Six steps because the census counts guard
# decisions, not trajectory statistics.
#
# The comparison per (model, rung, lever):
#   1. resolved changed            — a default flip, an import change, or a
#                                    threshold constant edited (the census
#                                    records the constant's VALUE, e.g.
#                                    TRANSPOSE_L1_RESIDENT resolves to 1.25);
#   2. fired fraction crosses zero — a dark lever starts firing, or a firing
#                                    lever goes dark (both directions fail:
#                                    a threshold quietly widening into a size
#                                    it was never measured at is a change too);
#   3. |frac - baseline| > 0.05    — partial darkness. K2 on main reads 560/0 at
#                                    512 and 560/560 at 768; a fired-SET
#                                    comparison calls that "still firing" and
#                                    passes, the fraction rule fails it;
#   4. a dark-and-ON lever with no one-line exemption reason in the baseline is
#      a FAIL, not a pass by silence;
#   5. setlen levers (SDPA_Q_CHUNK_FITS) gate the overflow-set size exactly —
#      every member is a fold that silently took the slow path.
#
# The exponent check uses runtime_s from the fold's own results.json, which
# excludes model load and process startup (the subprocess wall is
# load-dominated: measured k=0.35 vs the true k=1.09 over 128->512). runtime_s
# still carries a ~3 s size-independent term, which biases k DOWNWARD, so a
# recorded cliff is a lower bound on the true exponent. Tolerance per interval:
#
#     tol = max(0.50, 3 * sqrt(2) * sigma / ln(N2/N1))
#
# with sigma the measured relative noise of one rung's runtime_s (record mode
# measures it with SIZE_LADDER_SIGMA_REPS reps at 512 aa; boltz2 reads 6.5% on
# pc card 0, giving tol 0.50 on 256->512 and 0.68 on 512->768 against a
# measured cliff signal of 1.4-1.6). The 0.50 floor keeps a suspiciously quiet
# model from getting an unfalsifiably tight band. Above sigma = 12% the arm
# switches the model to median-of-3 (the repo's own answer to single-shot
# noise, merged 7431d6e39); if even that leaves the 512->768 tolerance wider
# than the ~1.4 cliff signal, exponents are recorded as SKIPPED for that model
# with the measured numbers as the reason — a measured "cannot be gated
# cheaply" beats a flaky arm.
#
# One warm-up fold at the smallest rung per model precedes the ladder: the
# first fold of a session reads ~24% slow (JIT + program caches), larger than
# the noise floor, and a cold first rung biases every exponent downward.
#
# Baselines are PER CARD TYPE (cards.<board_type>.models...), same discipline
# as docs/perf_baselines.json: the issue #11 fix scales L1 budgets to the
# part's measured per-core L1, so a p300c legitimately fires a different lever
# set than a p150a at the same sequence length. A card type with no recorded
# baseline is a loud NO BASELINE failure, never a silent skip. The exponent is
# a runtime RATIO, so it transfers across same-type machines to first order
# (qb1's p150a reads ~30% slower than pc's in absolute terms; the ratio cancels
# it); the census counts are shape/config-determined and machine-independent.
SIZE_LADDER_MODELS = ("boltz2", "esmfold2", "protenix-v2", "openfold3", "opendde")
SIZE_LADDER_RUNGS = tuple(int(x) for x in
                          os.environ.get("RELEASE_GATE_SIZE_RUNGS", "256,512,640,768").split(",")
                          if x.strip())
# Exponent intervals are taken over these rungs only; every other rung is
# census-only. 640 is in the ladder but not here on measured grounds: at the
# sigma = 6.5% noise floor a 3-sigma band over ln(640/512) = 0.223 is +-1.24 and
# over ln(768/640) = 0.182 is +-1.51, both at or past the ~1.40 cliff signal, so
# an exponent gate on those two intervals is a coin flip. Splitting 512->768
# into two ungateable halves would also destroy the one interval that IS
# gateable (+-0.68 over ln(768/512) = 0.405). So 640 earns its place as a lever
# rung and stays out of the timing chain.
SIZE_LADDER_EXP_RUNGS = (256, 512, 768)
SIZE_LADDER_BASELINE = REPO_ROOT / "docs" / "size_ladder_baseline.json"
SIZE_LADDER_STEPS = 6
SIZE_LADDER_FRAC_TOL = 0.05
SIZE_LADDER_SIGMA_REPS = 5
SIZE_LADDER_EXP_TOL_FLOOR = 0.50
# The diluted N^2 -> N^3.6 cliff reads as an apparent exponent jump of ~1.4 once
# the size-independent term in runtime_s biases both exponents downward. A 3-sigma
# tolerance wider than that cannot catch the cliff, so the model's exponent check
# is skipped (with the measured numbers) instead of shipping a coin flip.
SIZE_LADDER_EXP_MAX_TOL = 1.40
# Transient fold/census scratch; deleted after the run unless --keep. Lives under
# perf/sizegate, never the repo root (the 08-13 run_*.sh lesson).
SIZE_LADDER_WORKDIR = REPO_ROOT / "perf" / "sizegate" / "work"
# Record mode keeps the per-rung census artifacts here as the evidence behind
# docs/size_ladder_baseline.json — the first thing to diff when the arm goes red.
SIZE_LADDER_PROVENANCE = REPO_ROOT / "perf" / "sizegate" / "baseline"


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


def _repo_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=REPO_ROOT, text=True, timeout=5).strip()
    except Exception:
        return "unknown"


def _size_ladder_card_type() -> str:
    """Board-type key ('p150a', 'p300c', ...) for the per-card baseline lookup,
    reusing perf_regression's detector (tt-smi first, sysfs fallback; opens no
    device) so both gates key their baselines identically."""
    path = REPO_ROOT / "scripts" / "perf_regression.py"
    spec = importlib.util.spec_from_file_location("tt_bio_perf_regression", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.detect_card_type()


def _run_census_fold(model: str, rung: int, workdir: Path, tag: str) -> dict:
    """One lever-census-wrapped fold of the cdk2x2_<rung> fixture. Returns
    {"levers": {flag: {resolved, served, declined, frac, how}}, "runtime_s": ...,
    "wall": ...} or {"error": ...}.

    scripts/lever_census.py wraps the predict CLI so every spawned worker dumps
    its lever counters (predict folds in spawned processes, so counters read in
    the launcher are always zero); the fold itself is the arm's cheap config
    (single-sequence, 6 steps, 1 sample, seed 0) — enough to resolve every guard
    without paying for a production fold. runtime_s comes from the fold's own
    results.json, which excludes model load and process startup.
    """
    from tt_bio.main import predict_results_dir_name
    fixture = REPO_ROOT / "perf" / "size512" / "fixtures" / f"cdk2x2_{rung}.yaml"
    if not fixture.exists():
        return {"error": f"missing size-ladder fixture {fixture}"}
    label = f"{model}-{rung}-{tag}"
    census_json = workdir / f"census_{label}.json"
    out_dir = workdir / f"out_{label}"
    log = workdir / f"{label}.log"
    shutil.rmtree(out_dir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "lever_census.py"),
        "--tt-bio", sys.executable, "--label", label, "--out", str(census_json),
        "--", "-m", "tt_bio.main", "predict", str(fixture),
        "--model", model,
        "--single_sequence",
        "--sampling_steps", str(SIZE_LADDER_STEPS),
        "--diffusion_samples", "1",
        "--seed", str(SEED),
        "--out_dir", str(out_dir),
    ]
    t0 = time.monotonic()
    with open(log, "w") as fp:
        rc, timed_out = _run_fold(cmd, FOLD_TIMEOUT_S, cwd=REPO_ROOT,
                                  stdout=fp, stderr=subprocess.STDOUT)
    wall = time.monotonic() - t0
    if timed_out:
        return {"error": f"census fold timed out after {FOLD_TIMEOUT_S}s"}
    if rc != 0:
        tail = " / ".join(log.read_text(errors="replace").strip().splitlines()[-3:])[:200]
        return {"error": f"census fold exited {rc}: {tail}"}
    try:
        census = json.loads(census_json.read_text())
    except Exception as e:
        return {"error": f"census artifact unreadable: {e}"}
    levers = {}
    for r in census["rows"]:
        served, declined = r["served"], r["declined"]
        total = (served or 0) + (declined or 0)
        # never-reached (0/0) reads as frac 0.0 = dark; not-imported stays None
        frac = (served / total) if total else (0.0 if served == 0 else None)
        levers[r["flag"]] = {"resolved": r["resolved"], "served": served,
                             "declined": declined, "frac": frac, "how": r["how"]}
    results = out_dir / predict_results_dir_name(model, fixture.stem) / "results.json"
    runtime_s = None
    if results.exists():
        try:
            rows = json.loads(results.read_text())
            ts = [row["runtime_s"] for row in rows
                  if row.get("status") == "ok" and row.get("runtime_s") is not None]
            runtime_s = max(ts) if ts else None
        except Exception:
            runtime_s = None
    if runtime_s is None:
        return {"error": f"no runtime_s in {results.name} (fold ok but timing missing)"}
    return {"levers": levers, "runtime_s": runtime_s, "wall": wall,
            "census_json": census_json}


def _size_ladder_dark(entry: dict) -> bool:
    """A lever counts as dark when it resolved ON yet served no call. setlen
    levers have no served counter; their dark state is a non-empty overflow set.
    not-imported / False / off-by-design levers are absent, not dark."""
    if entry["resolved"] in ("False", "not-imported", "MISSING", "None", ""):
        return False
    if entry.get("how") == "setlen":
        return bool(entry.get("declined"))
    return entry.get("served") == 0


def _size_ladder_exemption_findings(base_levers: dict, where: str) -> list:
    """Every dark-and-ON lever in the baseline must carry a one-line reason."""
    findings = []
    for flag, b in base_levers.items():
        if _size_ladder_dark(b):
            reason = (b.get("reason") or "").strip()
            if not reason or reason.startswith("TODO"):
                findings.append(f"{where} {flag}: dark (resolved {b['resolved']}, "
                                f"served {b['served']}, declined {b['declined']}) with no "
                                f"exemption reason in the baseline")
    return findings


def _size_ladder_compare_levers(base: dict, cur: dict, where: str) -> list:
    """Findings comparing one rung's lever census against the baseline. The rules
    are the five in the SIZE_LADDER comment block above."""
    findings = []
    for flag, b in base.items():
        c = cur.get(flag)
        if c is None:
            findings.append(f"{where} {flag}: in baseline but missing from the census "
                            f"(lever removed? re-record the baseline)")
            continue
        if c["resolved"] != b["resolved"]:
            findings.append(f"{where} {flag}: resolved '{b['resolved']}' -> "
                            f"'{c['resolved']}' (default or threshold constant changed)")
            continue
        if b.get("how") == "setlen":
            if (c["declined"] or 0) != (b["declined"] or 0):
                findings.append(f"{where} {flag}: overflow-set size "
                                f"{b['declined']} -> {c['declined']}")
            continue
        fb, fc = b["frac"], c["frac"]
        if fb is None or fc is None:
            continue  # not-imported; the resolved equality above gates import drift
        if (fb == 0.0) != (fc == 0.0):
            findings.append(f"{where} {flag}: frac {fb:.3f} -> {fc:.3f} "
                            f"({'went dark' if fc == 0.0 else 'started firing'})")
        elif abs(fc - fb) > SIZE_LADDER_FRAC_TOL:
            findings.append(f"{where} {flag}: frac {fb:.3f} -> {fc:.3f} exceeds the "
                            f"{SIZE_LADDER_FRAC_TOL} band (partial darkness)")
    for flag in cur:
        if flag not in base:
            findings.append(f"{where} {flag}: new lever not in the baseline "
                            f"(re-record with --size-ladder-record)")
    return findings


def _size_ladder_measure_model(model: str, rungs, workdir: Path,
                               reps_512: int, reps_other: int) -> dict:
    """Warm up once (the first fold of a session reads ~24% slow, larger than the
    noise floor), then census-fold every rung. Returns {"levers": {rung: ...},
    "runtime_s": {rung: median}, "sigma": relative runtime noise at 512 | None,
    "census_jsons": {rung: path}} or {"error": ...}."""
    warm = _run_census_fold(model, rungs[0], workdir, "warmup")
    if warm.get("error"):
        return {"error": f"warm-up: {warm['error']}"}
    levers, runtimes, census_jsons = {}, {}, {}
    sigma = None
    for rung in rungs:
        reps = reps_512 if rung == 512 else reps_other
        runs = []
        for rep in range(reps):
            r = _run_census_fold(model, rung, workdir, f"rep{rep}")
            if r.get("error"):
                return {"error": f"rung {rung} rep {rep}: {r['error']}"}
            runs.append(r)
        counts = [{f: (l["served"], l["declined"]) for f, l in r["levers"].items()}
                  for r in runs]
        if any(c != counts[0] for c in counts[1:]):
            print(f"  [size-ladder] WARNING: {model}/{rung} census counts differ "
                  f"across reps — counts were assumed deterministic", flush=True)
        levers[str(rung)] = runs[0]["levers"]
        census_jsons[str(rung)] = runs[0]["census_json"]
        ts = [r["runtime_s"] for r in runs]
        runtimes[str(rung)] = round(statistics.median(ts), 2)
        if rung == 512 and len(ts) > 1:
            sigma = statistics.stdev(ts) / statistics.mean(ts)
    return {"levers": levers, "runtime_s": runtimes, "sigma": sigma,
            "census_jsons": census_jsons}


def _size_ladder_exponent_block(runtimes: dict, sigma):
    """Baseline exponent entries per consecutive rung pair: k with a tolerance
    derived from the measured noise floor. Returns (block, skip_reason)."""
    rungs = sorted(int(r) for r in runtimes if int(r) in SIZE_LADDER_EXP_RUNGS)
    if len(rungs) < 2:
        return None, "single rung — no interval to exponent over"
    if sigma is None:
        return None, "no noise measurement (rung 512 absent from the ladder)"
    reps, sigma_eff = 1, sigma
    if sigma > 0.12:
        # median-of-3, the repo's standing answer to single-shot noise
        # (perf-gate-single-shot-legs-recurring-false-alarm, merged 7431d6e39)
        reps, sigma_eff = 3, sigma / math.sqrt(3)
    exps = {}
    for n1, n2 in zip(rungs, rungs[1:]):
        k = math.log(runtimes[str(n2)] / runtimes[str(n1)]) / math.log(n2 / n1)
        tol = max(SIZE_LADDER_EXP_TOL_FLOOR,
                  3 * math.sqrt(2) * sigma_eff / math.log(n2 / n1))
        exps[f"{n1}->{n2}"] = {"k": round(k, 3), "tol": round(tol, 3)}
    worst = max(e["tol"] for e in exps.values())
    if worst > SIZE_LADDER_EXP_MAX_TOL:
        return None, (f"measured sigma {sigma:.1%} needs a ±{worst:.2f} band, wider "
                      f"than the ~{SIZE_LADDER_EXP_MAX_TOL} cliff signal — an exponent "
                      f"gate would be a coin flip for this model")
    return {"reps": reps, "sigma_runtime_512": round(sigma, 4), "exponents": exps}, None


def _size_ladder_fill_reasons(levers: dict, old_levers: dict) -> int:
    """Carry exemption reasons forward from the previous baseline; newly dark
    levers get a TODO. Returns the number of dark levers still needing a reason."""
    todo = 0
    for rung, table in levers.items():
        for flag, e in table.items():
            if not _size_ladder_dark(e):
                e.pop("reason", None)
                continue
            old = ((old_levers or {}).get(rung, {}).get(flag, {})).get("reason", "")
            if old and not old.startswith("TODO"):
                e["reason"] = old
            else:
                e["reason"] = ("TODO: one line on why this lever is legitimately "
                               "dark at this size")
                todo += 1
    return todo


def _size_ladder_check_model(model: str, rungs, base_model: dict, workdir: Path) -> dict:
    reps = base_model.get("reps", 1)
    meas = _size_ladder_measure_model(model, rungs, workdir, reps, reps)
    if meas.get("error"):
        return {"model": model, "gate": False, "error": meas["error"],
                "findings": [meas["error"]]}
    findings = []
    for rung in rungs:
        b_levers = base_model.get("levers", {}).get(str(rung))
        where = f"{model}/{rung}"
        if b_levers is None:
            findings.append(f"{where}: rung not recorded in the baseline")
            continue
        findings.extend(_size_ladder_exemption_findings(b_levers, where))
        findings.extend(_size_ladder_compare_levers(b_levers, meas["levers"][str(rung)],
                                                    where))
    measured_k = {}
    for interval, be in (base_model.get("exponents") or {}).items():
        n1, n2 = (int(x) for x in interval.split("->"))
        t1 = meas["runtime_s"].get(str(n1))
        t2 = meas["runtime_s"].get(str(n2))
        if t1 is None or t2 is None:
            continue  # custom RELEASE_GATE_SIZE_RUNGS narrower than the baseline
        k = math.log(t2 / t1) / math.log(n2 / n1)
        measured_k[interval] = round(k, 3)
        if abs(k - be["k"]) > be["tol"]:
            findings.append(f"{model} {interval}: exponent {be['k']:.2f} -> {k:.2f} "
                            f"outside ±{be['tol']:.2f}")
    return {"model": model, "gate": not findings,
            "error": "; ".join(findings) or None, "findings": findings,
            "runtime_s": meas["runtime_s"], "exponents": measured_k}


def run_size_ladder(keep: bool, record: bool, baseline_path: Path,
                    models=None) -> dict:
    """The size-generality arm (see the SIZE_LADDER comment block and the module
    docstring for the standing rule it enforces).

    Check mode folds every model at every rung through the lever census and
    fails on any divergence from the checked-in baseline. Record mode
    (--size-ladder-record) re-measures the baseline for THIS card type,
    preserving other cards' blocks and carrying existing exemption reasons
    forward; dark levers with no carried reason are written as TODO and the
    check mode refuses to pass until each carries a real one-line reason.
    """
    models = list(models or SIZE_LADDER_MODELS)
    rungs = SIZE_LADDER_RUNGS
    card = _size_ladder_card_type()
    workdir = SIZE_LADDER_WORKDIR
    baseline = {}
    if baseline_path.exists():
        try:
            baseline = json.loads(baseline_path.read_text())
        except Exception as e:
            return {"model": "size-ladder", "seconds": 0, "gate": False, "card": card,
                    "error": f"baseline {baseline_path} unreadable: {e}", "legs": []}
    if not record and not baseline:
        return {"model": "size-ladder", "seconds": 0, "gate": False, "card": card,
                "error": f"no baseline at {baseline_path} — record one with "
                         f"--size-ladder-record", "legs": []}

    print(f"\n{'='*70}\n[size-ladder] {'RECORDING baseline' if record else 'checking'} "
          f"for card {card}: {', '.join(models)} at rungs "
          f"{','.join(map(str, rungs))} ({SIZE_LADDER_STEPS} steps, 1 sample, "
          f"seed {SEED}, single-sequence)\n{'='*70}", flush=True)
    t0 = time.monotonic()
    legs = []
    if record:
        old_models = baseline.get("cards", {}).get(card, {}).get("models", {})
        # Seeded with the card's existing models, not empty: recording a subset
        # (--size-ladder-models) then UPDATES those models and leaves the rest of
        # the card block intact. A 5-model record is ~40 min of device time, so it
        # has to be resumable a model at a time instead of all-or-nothing.
        new_card = {"recorded": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
                    "commit": _repo_commit(), "models": dict(old_models)}
        todos = 0

        def _flush_baseline():
            baseline.setdefault("cards", {})[card] = new_card
            baseline.update({
                "format": 1,
                "what": "size-ladder release-gate baseline: per-model lever census and "
                        "runtime scaling exponents at every rung, per card type",
                "rule": "a perf lever may not land default-ON on the strength of one "
                        "sequence length; re-record after any size-affecting change",
                "record_with": "python3 scripts/release_gate.py --model size-ladder "
                               "--size-ladder-record",
                "rungs": list(rungs),
                "fold": {"single_sequence": True, "sampling_steps": SIZE_LADDER_STEPS,
                         "diffusion_samples": 1, "seed": SEED},
            })
            baseline_path.parent.mkdir(parents=True, exist_ok=True)
            baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")

        for m in models:
            meas = _size_ladder_measure_model(m, rungs, workdir,
                                              SIZE_LADDER_SIGMA_REPS, 1)
            if meas.get("error"):
                legs.append({"model": m, "gate": False, "error": meas["error"],
                             "findings": [meas["error"]]})
                continue
            block, skip = _size_ladder_exponent_block(meas["runtime_s"], meas["sigma"])
            todos += _size_ladder_fill_reasons(meas["levers"],
                                               old_models.get(m, {}).get("levers"))
            entry = {"runtime_s": meas["runtime_s"], "levers": meas["levers"]}
            if block:
                entry.update(block)
            else:
                entry["exponents_skipped"] = skip
            new_card["models"][m] = entry
            legs.append({"model": m, "gate": True, "error": None, "findings": [],
                         "runtime_s": meas["runtime_s"],
                         "exponents": {k: v["k"] for k, v in block["exponents"].items()}
                         if block else {}})
            if skip:
                legs[-1]["exponents_skipped"] = skip
            SIZE_LADDER_PROVENANCE.mkdir(parents=True, exist_ok=True)
            for rung, cj in meas["census_jsons"].items():
                shutil.copy(cj, SIZE_LADDER_PROVENANCE / f"census_{m}_{rung}_{card}.json")
            # After every model, so a run that dies at model 4 keeps models 1-3.
            _flush_baseline()
        _flush_baseline()
        if todos:
            print(f"[size-ladder] {todos} dark lever(s) need a one-line exemption "
                  f"reason — search TODO in {baseline_path} and fill them in; the "
                  f"check FAILS on any dark lever without a reason.", flush=True)
    else:
        card_block = baseline.get("cards", {}).get(card)
        if card_block is None:
            return {"model": "size-ladder", "seconds": time.monotonic() - t0,
                    "gate": False, "card": card,
                    "error": f"NO BASELINE for card type '{card}' in "
                             f"{baseline_path.name} — record one on this card type: "
                             f"--model size-ladder --size-ladder-record", "legs": []}
        for m in models:
            base_model = card_block.get("models", {}).get(m)
            if base_model is None:
                err = f"model not in the {card} baseline (added to " \
                      f"SIZE_LADDER_MODELS after recording? re-record)"
                legs.append({"model": m, "gate": False, "error": err,
                             "findings": [err]})
                continue
            legs.append(_size_ladder_check_model(m, rungs, base_model, workdir))

    gate = bool(legs) and all(l["gate"] for l in legs)
    row = {"model": "size-ladder", "seconds": time.monotonic() - t0, "gate": gate,
           "card": card,
           "error": next((l["error"] for l in legs if l["error"]), None),
           "legs": legs}
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return row


def main() -> int:
    # Scorers, folds and predict CLIs we spawn arm their parent-death guard off this,
    # so none of them can outlive this driver still holding a card. Inherited through
    # every spawn path in this file (see tt_bio/device_lease.py:arm_orphan_guard).
    os.environ["TT_BIO_PARENT_PID"] = str(os.getpid())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    choices=list(MODELS) + ["boltzgen", "opendde-abag", "capacity",
                                            "size-ladder"]
                    + ESMC_DEFAULT + ESMC_OPT_IN,
                    action="append",
                    help="Gate only this model (repeatable). Default: the five fold "
                         "models + boltzgen + opendde-abag + capacity + size-ladder + "
                         "ESMC 300m/600m embed parity. "
                         "esmc-6b is opt-in (slow ~13 GB load).")
    ap.add_argument("--keep", action="store_true", help="Keep run output dirs for inspection.")
    ap.add_argument("--size-ladder-record", action="store_true",
                    help="Re-record the size-ladder baseline for THIS card type instead "
                         "of checking against it. Explicit human action after an "
                         "intentional size-affecting change; runs all rungs and "
                         "measures the runtime noise floor. Never automatic.")
    ap.add_argument("--size-ladder-baseline", default=str(SIZE_LADDER_BASELINE),
                    help="Baseline JSON path (default docs/size_ladder_baseline.json).")
    ap.add_argument("--size-ladder-models", default=None,
                    help="Comma-separated subset of SIZE_LADDER_MODELS for a debug or "
                         "demo run (default: all five).")
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

    models = args.model or list(MODELS) + ["boltzgen", "opendde-abag", "capacity",
                                           "size-ladder"] + ESMC_DEFAULT
    fold_models = [m for m in models if m in MODELS]
    want_boltzgen = "boltzgen" in models
    want_opendde_abag = "opendde-abag" in models
    want_capacity = "capacity" in models
    want_size_ladder = "size-ladder" in models
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

    if want_size_ladder:
        sl = run_size_ladder(args.keep, args.size_ladder_record,
                             Path(args.size_ladder_baseline),
                             args.size_ladder_models.split(",")
                             if args.size_ladder_models else None)
        all_pass &= sl["gate"]
        rungs = SIZE_LADDER_RUNGS
        print(f"\n{'#'*78}\nRELEASE GATE — size-ladder (rungs "
              f"{','.join(map(str, rungs))}, {SIZE_LADDER_STEPS} steps / 1 sample, "
              f"seed {SEED}, single-sequence, card {sl.get('card', '?')})"
              + ("  [RECORD]" if args.size_ladder_record else "") + f"\n{'#'*78}")
        hdr = f"{'model':<15}" + "".join(f"{str(n) + 'aa':>9}" for n in rungs)
        hdr += "".join(f"{f'k{a}->{b}':>11}" for a, b in zip(rungs, rungs[1:]))
        hdr += f"{'wall':>9}  result"
        print(hdr)
        for l in sl["legs"]:
            rt = l.get("runtime_s") or {}
            ex = l.get("exponents") or {}
            cells = "".join(f"{(f'{rt[str(n)]:.1f}s' if rt.get(str(n)) is not None else '-'):>9}"
                            for n in rungs)
            for a, b in zip(rungs, rungs[1:]):
                k = ex.get(f"{a}->{b}")
                cells += f"{(f'{k:.2f}' if k is not None else '-'):>11}"
            wall = f"{sl['seconds']:.0f}s" if sl.get("seconds") is not None else "-"
            verdict = ("PASS" if l["gate"] else f"FAIL ({l['error']})" if l["error"]
                       else "FAIL")
            print(f"{l['model']:<15}{cells}{wall:>9}  {verdict}")
            for f in (l.get("findings") or []):
                print(f"    FAIL {f}")
        if not sl["legs"] and sl.get("error"):
            print(f"    FAIL {sl['error']}")
        print(f"{'#'*78}")
        if args.size_ladder_record:
            print("BASELINE RECORDED — fill every TODO exemption reason in "
                  f"{args.size_ladder_baseline}, then re-run without --size-ladder-record"
                  if sl["gate"] else
                  "BASELINE RECORD FAILED — a model did not fold (see above)")
        else:
            print("GATE PASS — lever census and scaling exponents match the recorded "
                  "baseline at every rung" if sl["gate"] else
                  "GATE FAIL — size-ladder drift vs the recorded baseline (see above); "
                  "if this change is intentional, re-record with --size-ladder-record")

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
