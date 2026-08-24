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

RFD3 is a design model too, but designability does not transfer to it. Upstream
RFdiffusion3 evaluates with ProteinMPNN/LigandMPNN sequences plus AF3 rather than
its own sequence head, tt-bio ships no MPNN, and docs/rfd3-design.md already tells
users the built-in sequence is a starting point to redesign — so refolding it would
score the model working as intended against a floor no failure could exceed. Its leg
scores the delivered structure instead: strict parse, backbone geometry as a clean
rate over four designs (a design is clean when no CA-CA step exceeds the break
distance and the rest sit in the measured band), heavy-atom clashes, a real sequence
at the designed positions (zero UNK plus a minimum distinct-amino-acid count), and
byte-identical coordinates from a repeated seed in a fresh process. Geometry is a rate
and not an every-design bar because RFdiffusion-family models produce an occasional
broken backbone by design, the same call BOLTZGEN_MIN_PASS_RATE already makes. Every
geometry and sequence number is computed over the DESIGNED residues only, recovered by re-featurizing the spec on the host: RFD3
merges a binder's designed residues into the target's own chain, so a chain-level
number averages 70 generated residues against 50 copied ones and passes by dilution.
This is the leg RFD3's two escaped defects would have tripped (a sequence head
computed and then dropped, and tile sparsity behind a wrong-variable gate) — neither
the bit-exact featurizer leg in ``scripts/full_parity_gate.py`` nor the perf leg can
see the far end of the pipeline. Geometry reuses
``perf/wh-correctness/check_structure.py``'s measured bands and clash search.

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

RF3 is gated twice, and the second leg is not redundant. Its MODELS row folds 7ROA at
117 aa, which is where every fold model here is scored; the ``rf3-1024aa`` leg folds the
997 aa 7EIP anchor and gates the device fold's CA-RMSD to the deposited structure. The
defects that hide at length are size-specific by construction -- a token axis that stops
bucketing to 32, an L1 gate fitted at 512 aa going dark above 640, a fused-SDPA chunk that
declines off-lattice -- and none of them touch 117 aa. The size-ladder arm below already
watches perf across four rungs; this is the accuracy half of the same idea at one rung. It
gates the crystal rather than X (device vs reference) because X moves when the reference
cache is regenerated on another backend, and a release that fails for that reason fails for
the wrong one. Reuses ``scripts/rf3_port/accuracy_cell.py`` and the reference cache
committed beside it, so the leg computes no reference and needs no GPU -- one device
rollout, ~4 min.

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
arm: fold each model at 256/512/640/768 aa through ``scripts/lever_census.py``
and fail when the fired/dark lever set or the log-log runtime exponent between
rungs drifts from the checked-in baseline (``docs/size_ladder_baseline.json``).
640 is the off-lattice rung: the other three all have a padded length the SDPA
chunk size divides, so a ladder of 256-multiples cannot see the class of defect
where the fused kernel declines at 448/576/640/704/832/896/960 only.
It is a change detector, not a purity check: legitimately dark levers carry a
one-line exemption reason in the baseline, and re-recording
(``--size-ladder-record``) is an explicit human action that runs every rung, so
flipping a default forces measuring four sizes. Baselines are per card type,
same discipline as ``docs/perf_baselines.json`` — a resource figure calibrated
on one board type and shipped everywhere is this same rule on the other axis,
so state a constant's board validity where you state its size validity. See
docs/size-generality.md.

    # gate everything (five fold models + BoltzGen designability + ESMC embed parity
    # + OpenDDE-abag docking + rf3 at 997 aa + capacity + size-ladder) on card 1
    TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> ESM_ROOT=/path/to/esm \
        OPENDDE_DOCKQ_PYTHON=/path/to/dockq_venv/bin/python \
        python scripts/release_gate.py
    # one leg
    python scripts/release_gate.py --model protenix-v2
    python scripts/release_gate.py --model boltzgen
    python scripts/release_gate.py --model esmc-300m
    python scripts/release_gate.py --model opendde-abag
    python scripts/release_gate.py --model rf3-1024aa
    python scripts/release_gate.py --model capacity
    python scripts/release_gate.py --model size-ladder
    # re-record the size-ladder baseline after an intentional size-affecting change
    python scripts/release_gate.py --model size-ladder --size-ladder-record

Exit code 0 iff every requested model PASSES its gate; 1 otherwise. Runs on the
device serially (one card context per run); no CPU shortcut for the fold/design.

The l1-budget leg is the only one that gates a PART, not a number. Every L1-edge budget
in ``tt_bio/tenstorrent.py`` was fitted on a 130-core p150a, and ``_apply_grid_thresholds``
returns early on any grid of 110 cores or more, so a 110-core Blackhole (p300/p300c) runs
budgets fitted for 130 cores and the trimul in-projection's circular buffers stop fitting
beside the pair tensors (issue #11, Taylor Singletary). The legs above cannot see that: they
compare numbers, and a part that dies at program creation produces no numbers. This leg runs
the budget arithmetic for every part class we have measured resource figures for, and folds
the input that died on his p300c across this part's grid ladder. THE RULE it enforces:
a part-specific resource figure entering ``tenstorrent.py`` gets a row in ``L1_BUDGET_PARTS``
in the same commit. Per-core L1 was the first such figure; total DRAM is the second, and it
carries the host-concat byte budget.

This is the *accuracy* leg of the release gate. The *UX* leg lives in
``scripts/ux_regression.py`` (live-progress phases, output parsing, CLI shape) —
see RELEASING.md. The two are independent; both must exit 0 before a tag.
"""

import argparse
import csv
import hashlib
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import gate_guard  # noqa: E402  (host-load guard, shared with full_parity_gate.py)

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
# One model does not fold at the shared step count, and it is not a preference: 200 is not a
# configuration RF3 ships for inference. Its inference engine sets num_steps: 50
# (tt_bio.main._resolve_sampling_steps cites the upstream config and the checkpoint's
# training-side 200 that the engine overrides), and every RF3 parity and perf number in this
# repo was produced at 50. Folding the gate at 200 would gate a config no user reaches by
# default, the same reason _msa_args folds esmfold2 single-sequence.
SAMPLING_STEPS_BY_MODEL = {"rf3": 50}
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
    # RoseTTAFold3, shipped as `predict --model rf3` since v0.6.6. Measured on this gate
    # 2026-08-23 on tt-quietbox card 0 (p150a, 13x10 grid, MSA on, rf3's own 50 sampling
    # steps): RMSD 1.238 A / TM 0.958, the closest to the experimental structure of the seven
    # models here, in 216 s. Two separate processes at seed 0 agreed on all five samples to
    # the printed digit, so this floor covers seed and MSA-draw spread rather than a
    # run-to-run wobble that does not exist at 117 aa. Floor = ~2x measured, same discipline
    # as boltz2 (1.55 -> 3.0), protenix-v2 (3.87 -> 6.0) and openfold3 (1.775 -> 3.5).
    "rf3":           {"max_rmsd": 3.0, "min_tm": 0.75},
    # OpenBind-0, shipped as `predict --model openbind`. Same OF3 stack on upstream's
    # v0.5.0 checkpoint, so it folds the shared 7ROA target with the same MSA bytes the
    # openfold3 row above uses (/home/ttuser/ypx_msa/425f6b0c8f93f94f.a3m, sha256
    # 98eb9adc..., 35 sequences) and the two rows are comparable. Measured on this gate
    # 2026-08-23 on tt-quietbox2 card 1 (p150a, MSA on): RMSD 1.693 A / TM 0.894
    # best-confidence, 265 s; the five samples spread 1.589-2.183 A / 0.830-0.923, so this
    # floor covers sample spread and not a run-to-run wobble. Same numbers as openfold3's
    # 1.775 / 0.890 to within that spread, which is what "the v0.5.0 delta is worth nothing
    # here" looks like on a 117 aa target. Floor = ~2x measured, so the same pair openfold3
    # carries -- same discipline as boltz2 (1.55 -> 3.0) and protenix-v2 (3.87 -> 6.0).
    "openbind":      {"max_rmsd": 3.5, "min_tm": 0.70},
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

# --- rfd3 leg -----------------------------------------------------------------
# RFD3 is a design model with no ground truth and no usable designability metric: upstream
# RFdiffusion3 evaluates with ProteinMPNN/LigandMPNN sequences + AF3, not its own sequence
# head, tt-bio ships no MPNN, and docs/rfd3-design.md already tells users the built-in
# sequence is a starting point to redesign. Refolding it would score the model working as
# intended (43-60% ALA de novo) against a floor no failure could exceed.
#
# So this leg gates what RFD3's other legs cannot see. The featurizer leg
# (full_parity_gate.py's "rfd3-featurizer", 43/43 f keys bit-exact) scores an INPUT and the
# perf leg scores a wall-clock; both stay green when the design leaving the far end is
# garbage. Both of RFD3's escaped defects lived exactly there: the sequence was computed and
# then dropped (fixed 16dfe4db), and a wrong-variable gate shipped tile sparsity. This leg
# scores the delivered CIF instead: backbone geometry, no clashes, a real sequence at the
# designed positions, and byte-identical output from a repeated seed.
#
# It scores the DESIGNED RESIDUES ONLY, recovered by re-featurizing the spec on the host and
# reading restype == DESIGNED_RESTYPE_IDX off the shipped featurizer. That is not optional:
# for a binder RFD3 merges the designed residues into the target's own chain (gate-binder is
# one chain A of 120 = 50 copied target + 70 designed), and once the sequence head has named
# them nothing in the CIF marks which is which. Chain-level scoring would average 70
# generated residues against 50 copied ones, the pass-by-dilution that let the all-UNK defect
# through two of three protocols.
RFD3_SPEC = REPO_ROOT / "examples" / "rfd3_binder.json"
RFD3_SPEC_ID = "gate-binder"
RFD3_NUM_DESIGNS = 4
RFD3_SEED = 42
# 200 steps, not the CLI default of 4: docs/rfd3-design.md calls 4 a fast smoke setting and
# 200 the upstream production default. A 4-step trajectory is barely denoised, so a geometry
# bar on it would gate a config nobody wants results from.
RFD3_TIMESTEPS = 200
# The determinism arm repeats one design in a fresh process. Bit-exactness is a property of
# the device graph and shows up on the first step, so 4 steps buys the same signal at ~1/50
# the cost of 200.
RFD3_DET_TIMESTEPS = 4

# Floors, each doubled the way its own metric's noise behaves. A flat "2x the measured
# number" is wrong for a fraction (2x an in-band fraction of 1.0 is 0.5, which nothing can
# fail), so a fraction gets 2x its DEFICIT, a count gets 2x itself, and a binary wiring fact
# gets no band at all.
#
# MEASURED on qb1 (Blackhole p150a, card 0), ttnn 0.67.4, commit f77f8ad1,
# examples/rfd3_binder.json at 200 steps x 4 designs, seed 42, scored over the 70 designed
# residues (per-design, not averaged):
#
#   design  CA-CA median  in band  breaks  clashes/heavy   distinct AA  UNK  top AA
#     0        3.868 A     1.0000     0       0/715             14        0   45.7%
#     1        3.865 A     0.9855     1       2/714             12        0   51.4%
#     2        3.891 A     1.0000     0       0/715             14        0   42.9%
#     3        3.869 A     1.0000     0       3/714             11        0   42.9%
#
# Geometry is gated as a CLEAN RATE, not as an all-designs bar, because the measurement says
# it has to be: design 1 carries one genuine 8.497 A CA-CA step at residue 58->59, while
# every other step across all four designs sits in 3.76-4.04 A. RFdiffusion-family models
# produce an occasional broken backbone and the field's answer is to generate several and
# filter, so "zero breaks in every design" is a model-quality bar, not a port-correctness
# one, and a gate set there would fail correct code roughly a quarter of the time. This is
# the same call the BoltzGen leg above already makes with BOLTZGEN_MIN_PASS_RATE, for the
# same stated reason: one bad seed out of four should not fail the gate, all four should.
# Measured clean rate 3/4 = 0.75, floor 0.50.
#
# The clean test is two-sided on purpose and neither half is redundant: `breaks` catches one
# severe discontinuity (design 1 still scores 0.9855 in band, which clears the 0.98 bar on
# its own), and `in_band` catches many mildly-wrong steps that never individually exceed
# CA_CA_BREAK. A garbled coordinate tensor trips one or the other.
RFD3_MIN_CLEAN_RATE = 0.50   # measured 0.75 (3 of 4)
RFD3_MIN_INBAND = 0.98       # measured 1.0000 / 0.9855 / 1.0000 / 1.0000
RFD3_MAX_BREAKS = 0          # per design, inside the clean test above
# Clashes stay an absolute count, worst design of four, against check_structure's own
# calibrated allowance (CLASH_MAX_ABS = 2, the worst measured experimental structure) doubled
# past what this fixture measured. Not a fraction: 2x a measured 0 is a zero-tolerance bar,
# and "0.004" hides that it is 3 atoms. A tile-sparsity blowup is tens to hundreds of clashes
# (the pre-fix WH artifact logged 53), so 6 catches that class with room to spare.
RFD3_MAX_CLASHES = 6         # measured worst 3
# Sequence: the designed positions must carry a real predicted sequence. Zero UNK is binary
# and gets no band -- it is the exact escaped defect (sequence head computed then dropped,
# fixed 16dfe4db), and the host-only negative control reproducing that writer scores 70 UNK
# and 0 distinct AA. The distinct-AA count is a second, weaker guard against the sequence
# head collapsing to one residue; deliberately NOT a composition cap, since 43-60% ALA is
# real RFD3 behaviour (measured 42.9-51.4% here) and a cap would fail a correct model.
RFD3_MIN_DISTINCT_AA = 5     # measured worst 11
RFD3_MAX_UNK = 0             # measured 0; the pre-fix writer scores 70
# Runs by default: the whole arm (4 designs at 200 steps in one batched forward, plus the two
# 4-step determinism repeats) measured 95 s and 108 s on two runs on qb1 card 3 with the host
# otherwise uncontended, well inside BoltzGen's ~271 s default-arm precedent. The same arm took 486 s
# earlier the same day only because every device open queued behind another worker on the
# host-wide /tmp/tt-bio-device-open.lock, which measures contention, not the leg.

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
# ...and bounded like every fold this gate launches, for the same reason: an unbounded wait on
# an external tool blocks the whole gate run, not just its own leg. DockQ scores the 1ahw
# fixture in ~1 s on this hardware, so 300 s is ~300x headroom and still fails loud in five
# minutes instead of hanging forever on a corrupt CIF or a wedged reference tool.
DOCKQ_TIMEOUT_S = int(os.environ.get("RELEASE_GATE_DOCKQ_TIMEOUT", "300"))

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
#      every member is a fold that silently took the slow path;
#   6. the set of decline CLAUSES changed. Six of these modules keep a
#      (reason, shape) reject dict and the census now records the reason, so a
#      guard that refuses for a clause it did not refuse for before is a
#      behaviour change with no fired-fraction and no timing signature —
#      nothing else in this arm can see it. It is also what makes an exemption
#      entry evidence: the TODO the recorder writes carries the measured clause
#      ("declines on m_tiles=64"), so the human is confirming a fact instead of
#      reconstructing one from source.
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
# The first fold AT EACH RUNG is discarded. The JIT kernel cache is keyed by
# shape, so folding 256 aa does not warm 512 aa, and the old policy (one warm-up
# at the smallest rung) left every other rung's first measured fold cold.
# Measured on esmfold2 at 512 aa, five reps after a 256 aa warm-up: 82.3, 47.9,
# 49.8, 51.0, 52.5 s -- keeping the first reads sigma 25.4% and a +-2.66 band,
# which records the model as ungateable; dropping it reads 3.9% and +-0.40,
# tighter than boltz-2. That policy was calibrated on boltz-2, where one fold is
# enough: the same one-size-fits-all mistake this arm exists to catch, in the arm.
# Cost is one extra fold per rung per model.
#
# Baselines are PER CARD TYPE (cards.<board_type>.models...), same discipline
# as docs/perf_baselines.json: the issue #11 fix scales L1 budgets to the
# part's measured per-core L1, so a p300c legitimately fires a different lever
# set than a p150a at the same sequence length. A card type with no recorded
# baseline is a loud NO BASELINE failure, never a silent skip. The exponent is
# a runtime RATIO, so it transfers across same-type machines to first order
# (qb1's p150a reads ~30% slower than pc's in absolute terms; the ratio cancels
# it). The census counts do NOT transfer as freely as an earlier version of this
# comment claimed: a guard sized against the core grid flips with the grid, and
# protenix-v2's K2 is admitted on 11x10 and refused on 13x10. Board type does not
# pin the grid either, because harvesting means one board type presents several.
# So each model's block records the grid its census ran on and the check REFUSES
# a cross-grid comparison outright instead of reporting levers as newly dark.
SIZE_LADDER_MODELS = ("boltz2", "esmfold2", "protenix-v2", "openfold3", "opendde",
                      "rf3", "nesso1", "openbind")
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

# nesso1 is the one model in this arm that `predict` cannot fold. It returns a scalar rather
# than a structure, so it ships as `tt-bio affinity` and sits in AFFINITY_MODELS, not
# PREDICT_MODELS. It needs its own leg because the shared cdk2x2 fixture leaves its whole
# forward path dark: four of the other five models have no affinity module, so that fixture is
# apo protein with no ligand, and NO rung of this arm exercised the affinity pairformer for any
# model. The measured consequence is not hypothetical — TRIATT_PERSISTENT_MASK serves 0 of 768
# calls per forward with affinity=True at every rung, because the pair-mask slice makes the bias
# per row ([S, h, S, S]) instead of batch-broadcast ([1, h, S, S]) and the fused kernel correctly
# declines. Read off the apo fixture the same lever looks fully served. A lever 100% dark on a
# shipped path is what this arm exists to catch.
#
# Fixtures: perf/nesso1/inputs/ladder/aa<rung>/cdk2_<rung>.yaml — the same tiled CDK2 sequence
# the other models' rungs use plus the upstream README ligand, so nesso1's rung at N aa is the
# same protein as everyone else's rung at N aa. 256 aa featurizes to 276 tokens (the ligand is
# tokenised per heavy atom).
#
# Fold config is every shipped default: --trunk bf16, --recycling_steps 5, --tokens_budget 256.
# The pocket crop is load-bearing for correctness on large targets, so gating anything else
# would gate a configuration nobody runs — and it is why nesso1's exponents are LOW: only the
# first of the six trunk passes runs at full N, the other five run at the crop. That dilutes a
# first-pass cliff rather than hiding it, and the recorded band is measured, not assumed.
# --single_sequence / --sampling_steps / --diffusion_samples do not apply: no MSA, no diffusion.
# Timing is the `seconds` column of affinity.csv, the forward alone, which is the same thing
# predict's runtime_s is.
#
# The checkpoint ships a 413 MB ccd.pkl that is never committed. Set NESSO_CACHE to the HF cache
# holding it when it is not in the default one; the leg fails naming the variable.
SIZE_LADDER_NESSO_RECYCLING = 5
SIZE_LADDER_NESSO_TOKENS_BUDGET = 256


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
# tt-bio selects on it and its total DRAM. THE RULE, and the reason this leg exists: a
# part-specific resource figure entering tenstorrent.py gets a row here in the same commit.
# DRAM is the second such figure: CONCAT_HOST_BYTES_BASE = 1.5 GiB is one eighth of the 12.0
# GiB the Wormhole chip it was measured on reports, and shipped unscaled it sent the OpenDDE
# refiner's pair channel join to the host on a 31.875 GiB part, 21.88x per call.
L1_BUDGET_PARTS = (
    # (name, grid, per-core unreserved L1 bytes, total DRAM bytes, provenance)
    ("p150a", (13, 10), 1532416, 34_225_520_128,
     "L1 measured on pc 2026-08-19, ttnn.get_max_worker_l1_unreserved_size(); DRAM measured "
     "on tt-quietbox card 0 2026-08-20, ttnn.get_memory_view (8 x 4278190016 = 31.875 GiB)"),
    ("p300c", (11, 10), 1532416, 34_225_520_128,
     "L1 measured on tt-quietbox2 device 0 2026-08-19, same call. Identical to p150a — the "
     "p300 difference that broke issue #11 is core count (110 vs 130), not per-core L1. DRAM "
     "measured there 2026-08-20, also 8 x 4278190016, so p300c and p150a share both budgets"),
    ("wormhole", (8, 8), 1466080, 12 * 2 ** 30,
     "_WH_MEASURED_L1_PER_CORE (the L1 the WH re-fit was measured at); 12.0 GiB DRAM is the "
     "Galaxy chip total dram_peak prints, and 12.0/8 is exactly CONCAT_HOST_BYTES_BASE"),
    ("wormhole-full", (8, 8), 1572864, 12 * 2 ** 30,
     "_WH_FULL_L1_PER_CORE (1.5 MiB, the WH scaling reference); same 12.0 GiB part"),
)

# The OpenDDE refiner pair tensor (H = 1.945 * aa, c_z = 384, bf16) at the two rungs that
# decided the host-concat budget. H=1494 is 1.5965 GiB, over a 12 GiB part's budget and under
# a 31.875 GiB part's; H=1243 is 1.1051 GiB, under both, so it is the negative control.
CONCAT_REFINER_SHAPES = ((768, 1494, True), (640, 1243, False))

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
# --- nesso1 correctness leg ------------------------------------------------------
# Nesso-1 folds nothing, so it cannot join MODELS: there is no structure to Kabsch and no
# TM to score. Its output is eleven scalars, and the only comparison available is against
# the torch reference, which is bit-exact against upstream. The arm delegates to
# scripts/nesso1_port/device_parity.py — the harness the port already validated with —
# the same way the opendde-abag leg delegates its DockQ to scripts/opendde_dockq.py.
# The floors live in that script, next to the measurement, so a hand run and the gate
# report the same verdict.
#
# The arm scored is the one that ships: `tt-bio affinity` defaults to a bf16 trunk and
# fp32 affinity stacks. Do not score bf16/bf16 — it is 3.17 rather than 3.43 on X_over_R
# and nobody runs it.
NESSO1_PARITY = REPO_ROOT / "scripts" / "nesso1_port" / "device_parity.py"
NESSO1_FIXTURE = REPO_ROOT / "scripts" / "nesso1_port" / "parity_artifacts" / "tyr48"
NESSO1_TRUNK, NESSO1_AFFINITY = "bf16", "fp32"
NESSO1_REPEATS = 3
# ~2-3 min: a CPU torch reference at 61 tokens is 8.5 s, three device repeats are ~1 s
# each, and the rest is the weight load.
NESSO1_TIMEOUT_S = 1800

# PXDesign design leg. PXDesign is a binder-design pipeline, so it has no row in MODELS (folds of
# 7ROA scored by CA-RMSD/TM against a ground truth) and none in SIZE_LADDER_MODELS -- neither holds
# a design model. Same shape as the BoltzGen designability leg: run the shipped CLI path, parse
# every CIF strictly, then score the one end-to-end number the model actually makes a claim about.
#
# That number is the conditioned-token fit RMSD. PXDesign sees a 64-bin DISTOGRAM of the target and
# nothing else about its coordinates, so a correct run reproduces the target's own fold while the
# binder is free, and a broken conditioning path (wrong embedding row, wrong bin edges, a leaked
# binder placeholder) lands in the tens of angstroms. Nothing else in the pipeline complains about
# any of those, which is what makes it the right gate rather than a nicety.
PXDESIGN_SPEC = REPO_ROOT / "tests" / "fixtures" / "pxdesign" / "PDL1.yaml"
PXDESIGN_N_STEP = 20        # steps change how often a site is reached, not which; keeps the leg short
PXDESIGN_SEED = 0
PXDESIGN_NUM_DESIGNS = 1
# Floor, not a target: catch a gross conditioning failure, the same philosophy as the MODELS floors.
# Measured 2026-08-23 on qb2 card 0 (Blackhole p300c) through the shipped CLI path at n_step=20,
# seed 0: 4.909 A over the 116 conditioned tokens of the 196-token PD-L1 fixture. The floor is 3x
# that. Generous on purpose -- the failure this catches is a BROKEN CONDITIONING PATH, which lands
# in the tens of angstroms, not a design that is a little worse. 20 steps is a smoke, not
# production quality: upstream uses 400, so this says the conditioning works, not that the binder
# is good.
PXDESIGN_MAX_FIT_RMSD = 15.0
PXDESIGN_FIT_RMSD_MEASURED = 4.909
# Bit-exactness evidence, REPORTED rather than gated. A coordinate digest is card- and
# arch-specific the way af2ig-trunk-device's floor is, so making release success turn on equality
# here would fail a release host for having different silicon rather than for a defect.
# Recorded alongside the floor above and reproduced across three runs of the leg.
PXDESIGN_STRUCTURE_SHA16 = "64af2cbc286012b9"

# --- rf3 997 aa accuracy leg ---------------------------------------------------
# Named for the size-ladder rung it sits on (1024 aa); the fixture is 7EIP at 997
# residues, the largest real target on that rung and deliberately not a multiple of 32.
#
# The MODELS row above gates rf3 on 7ROA, 117 aa. Nothing else here reads RF3's accuracy
# at the length a customer target actually has, and the defects that hide at length are
# size-specific by construction: a token axis that stops bucketing to 32, an L1 gate
# fitted at 512 aa that goes dark above 640, a fused-SDPA chunk that declines off-lattice.
# None of them touch 117 aa. The size-ladder arm already watches perf across four rungs;
# this is the accuracy half of the same idea, at one rung.
#
# It gates the CRYSTAL reading: the device fold's CA-RMSD to the deposited 7EIP structure.
# Not X (device vs reference), which is the accuracy cell's headline number, because X
# needs the reference half and therefore moves when the reference cache is regenerated on
# another backend -- a release that fails because the reference moved fails for the wrong
# reason. The crystal does not move.
#
# Reuses scripts/rf3_port/accuracy_cell.py, the harness the anchor was measured with, run
# as a subprocess the way the nesso1 leg delegates to its own parity harness: it opens a
# device context and this process must stay free for the arms after it. The reference
# coordinates come from the cache committed next to the cell, so the leg computes no
# reference and needs no GPU; it pays one device rollout.
RF3_1024AA_CELL = REPO_ROOT / "scripts" / "rf3_port" / "accuracy_cell.py"
RF3_1024AA_FIXTURE = "7eip_997"
RF3_1024AA_REF_CACHE = (REPO_ROOT / "perf" / "rf3" / "results"
                        / f"accuracy_{RF3_1024AA_FIXTURE}")
# One seed. The cell's five-seed run spread 1.7362-2.1523 A device-vs-crystal, a 0.42 A
# band under a 4.0 A floor, so seeds two through five buy nothing a floor can read and
# cost another device rollout each.
RF3_1024AA_SEED = 0
# Floor = ~2x measured, the same discipline every MODELS floor uses (boltz2 1.55 -> 3.0,
# openfold3 1.775 -> 3.5, rf3/7ROA 1.238 -> 3.0). Measured 1.9687 A at seed 0
# (perf/rf3/results/a0_7eip997.json) through the shipped arm: 10 recycles, RF3's own 50
# sampling steps, one diffusion sample, HiFi4 + fp32_dest_acc + packer_l1_acc, the MSA
# committed with the fixture. Catches a gross size-specific accuracy failure, not a fold
# that is slightly worse.
RF3_1024AA_MAX_XTAL_A = 4.0
RF3_1024AA_XTAL_MEASURED = 1.9687
# The leg measured 407 s end to end on qb2 card 3 with three other gate legs sharing the
# host: 23 s featurize, 368 s device rollout, the rest the reference checkpoint load the
# cell does before it knows whether it needs a reference. The same rollout measured 210 s
# on an uncontended host, so a release host should see ~4 min. Same 1800 s bound the fold
# legs carry, env-tunable for the same reason.
RF3_1024AA_TIMEOUT_S = int(os.environ.get("RELEASE_GATE_RF3_1024AA_TIMEOUT", "1800"))

DEFAULT_ARMS = ("boltzgen", "rfd3", "opendde-abag", "capacity",
                "l1-budget", "batch-position", "nesso1", "pxdesign",
                "rf3-1024aa")

L1_BUDGET_STEPS = 200
# The width measured to fit at the issue-#11 call shape on a 110-core grid: the
# clash-and-narrow fold lands here, so a fold pinned here from the start is the
# same computation without the interrupted attempt.
L1_BUDGET_CAP = 128

# --- batch-position leg --------------------------------------------------------
# 0.6.5 fixed a defect every other leg is structurally blind to: three byte-identical
# Boltz-2 affinity targets folded by ONE worker scored 0.648724 / 0.722511 / 0.687149,
# differing only by their position in the batch. Unseeded RDKit ETKDG redrew the ligand
# conformer on every parse, and the affinity checkpoint's lazy load advanced the RNG the
# first target's diffusion drew from. Every other leg folds a single target per process,
# so a per-position difference has nothing to differ from and cannot be seen.
# The arm folds N identical targets plus one genuinely different one in a single
# process. The identical ones must agree exactly; the different one must NOT, so the
# arm cannot pass by everything collapsing to a constant.
BATCH_POSITION_SCRIPT = REPO_ROOT / "scripts" / "boltz2_affinity_batch_position_repro.py"
BATCH_POSITION_N = 3
BATCH_POSITION_AA = 256
BATCH_POSITION_EXTRA_AA = 200
# The fields that must be position-independent: the structure the ligand conformer
# feeds, and both affinity heads.
BATCH_POSITION_FIELDS = ("coords", "affinity_pred_value", "affinity_probability_binary")


def _sampling_steps(model: str) -> int:
    """Requested diffusion steps for one model's gate fold (see SAMPLING_STEPS_BY_MODEL)."""
    return SAMPLING_STEPS_BY_MODEL.get(model, SAMPLING_STEPS)


def _steps_label(models) -> str:
    """The step count for a summary header over a table of models: one number when they all
    folded at the same one, "200 (rf3 50)" when they did not. A single number printed over a
    table where one row used another one is how a summary stops matching its own run."""
    counts = {m: _sampling_steps(m) for m in models}
    if len(set(counts.values())) <= 1:
        return str(next(iter(counts.values()), SAMPLING_STEPS))
    odd = ", ".join(f"{m} {c}" for m, c in counts.items() if c != SAMPLING_STEPS)
    return f"{SAMPLING_STEPS} ({odd})"


def _msa_args(model: str) -> list:
    """MSA source for one model's gate fold — the way that model is ACTUALLY used.

    ``tt_bio.main.MSA_DEFAULT_MODELS`` is the source of truth for which models resolve an MSA
    by default (boltz2 / protenix-v2 / openfold3 / openbind / opendde / opendde-abag): those fold
    with an offline cached-a3m dir if RELEASE_GATE_MSA_DIR is set, otherwise the ColabFold
    server (bounded by FOLD_TIMEOUT_S — see RELEASING.md). esmfold2 is single-sequence with an optional MSA and
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

    from tt_bio.cache import cached, seq_hash

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
            a3m = Path(MSA_DIR) / f"{seq_hash(seq)}.a3m"
            if not cached(a3m):
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


def _kill_group(proc) -> None:
    """Kill the whole process group of a timed-out child started with start_new_session.

    start_new_session makes the child a session/group leader, so its pgid is its pid.
    Escalate unconditionally: the direct child exiting is NOT evidence the group is clear.
    predict folds inside a multiprocessing spawn grandchild that survives SIGTERM while
    still holding /dev/tenstorrent/N, and breaking out of the escalation as soon as
    proc.wait() returned left exactly that orphan behind (2026-08-22: one hung 640 aa fold
    reparented a spawn child to init, and the two following legs both failed with
    "device-open failure: the card is leased by another process or wedged" — one hang
    cascading into spurious failures). The same shape covers the DockQ leg, whose
    OPENDDE_DOCKQ_PYTHON is a wrapper script on some hosts: killing only the wrapper leaves
    the interpreter it exec'd running.
    """
    import signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except Exception:
            pass              # group already gone; still send the next signal
        try:
            proc.wait(timeout=10)
        except Exception:
            pass


def _run_fold(cmd: list, timeout: float, **popen_kw) -> tuple:
    """Run a fold subprocess in its OWN process group; on timeout kill the whole group so a
    hung MSA-server wait or a hung multiprocessing shutdown cannot orphan device-holding
    children (which would wedge the card for later legs). Returns (returncode, timed_out)."""
    proc = subprocess.Popen(cmd, start_new_session=True, **popen_kw)
    try:
        return proc.wait(timeout=timeout), False
    except subprocess.TimeoutExpired:
        _kill_group(proc)
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
        "--sampling_steps", str(_sampling_steps(model)),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--seed", str(SEED),
        *_msa_args(model),
        "--out_dir", str(REPO_ROOT),
    ] + ((["--fast"] if FAST else [])
          + (["--diffusion_trace"] if (DIFFUSION_TRACE and model == "boltz2") else []))
    print(f"\n{'='*70}\n[{model}] folding {DATA.name} "
          f"({_sampling_steps(model)} steps, {DIFFUSION_SAMPLES} samples)\n{'='*70}",
          flush=True)

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


def _load_geometry_harness():
    """Import perf/wh-correctness/check_structure.py by path — reuse its measured band
    constants and its clash search, do not re-derive protein geometry here. That file is
    the validated structural checker the WH-correctness sweep scores every design with."""
    path = REPO_ROOT / "perf" / "wh-correctness" / "check_structure.py"
    spec = importlib.util.spec_from_file_location("tt_bio_check_structure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rfd3_designed_residues() -> set:
    """(chain, res_id) of every designed protein token, from the shipped host featurizer.

    Host-only, no device. The keys are the same ones _write_cif writes the CIF with
    (``_chain_label(asym_id[t])`` and ``residue_index[t]``), so this identifies the designed
    residues in the output file exactly and without relying on residue order.
    """
    import json as _json
    from tt_bio.rfd3.input import InputSpecification
    from tt_bio.rfd3.featurize import featurize, DESIGNED_RESTYPE_IDX
    from tt_bio.rfd3.design import _chain_label

    raw = _json.loads(RFD3_SPEC.read_text())[RFD3_SPEC_ID]
    spec = InputSpecification.from_dict(raw)
    f = featurize(str(REPO_ROOT / spec.input), spec)
    rt = f["restype"]
    rt = rt.argmax(-1) if rt.ndim == 2 else rt
    mask = (rt == DESIGNED_RESTYPE_IDX) & f["is_protein"].bool()
    asym, resid = f["asym_id"].tolist(), f["residue_index"].tolist()
    return {(_chain_label(int(asym[t])), int(resid[t]))
            for t in range(len(rt)) if bool(mask[t])}


def _rfd3_score_cif(cif: Path, designed: set, geom) -> dict:
    """Score one delivered RFD3 CIF over the designed residues only.

    Geometry uses ``geom``'s measured bands (CA_CA_BAND, CA_CA_BREAK, CLASH_DIST) and its
    clash search; the only thing computed here is the restriction to ``designed``, which
    ``geom.chain_geometry`` cannot express (it is chain-level, and the designed residues
    share the target's chain).
    """
    import gemmi
    import numpy as np

    st = gemmi.read_structure(str(cif))
    st.setup_entities()
    cas, resnames = [], []
    for chain in st[0]:
        for res in chain:
            if (chain.name, res.seqid.num) not in designed:
                continue
            a = res.find_atom("CA", "*")
            if a is None:
                continue
            cas.append((res.seqid.num, [a.pos.x, a.pos.y, a.pos.z]))
            resnames.append((res.seqid.num, res.name))
    cas.sort(key=lambda t: t[0])
    resnames.sort(key=lambda t: t[0])
    a = np.asarray([c for _, c in cas])
    if len(a) < 2:
        raise ValueError(f"{cif.name}: found {len(a)} designed CA of {len(designed)} expected")
    d = np.linalg.norm(a[1:] - a[:-1], axis=1)
    lo, hi = geom.CA_CA_BAND
    seq = [geom._one_letter(n) for _, n in resnames]
    n_clash, heavy, worst = geom.clashes(st)
    return {"n_designed": len(a),
            "in_band": round(float(((d >= lo) & (d <= hi)).mean()), 4),
            "step_median": round(float(np.median(d)), 3),
            "breaks": int((d > geom.CA_CA_BREAK).sum()),
            "clashes": n_clash, "heavy": heavy,
            "clash_frac": round(n_clash / heavy, 6) if heavy else 0.0,
            "worst_contact": worst,
            "unk": sum(1 for c in seq if c == "X"),
            "distinct_aa": len({c for c in seq if c != "X"}),
            "top_aa_frac": round(max(seq.count(c) for c in set(seq)) / len(seq), 3)}


def _rfd3_design_cmd(out: Path, num_designs: int, steps: int) -> list:
    return [sys.executable, "-m", "tt_bio.main", "design", str(RFD3_SPEC),
            "--model", "rfd3", "--from_pdb", "--out_dir", str(out),
            "--num_designs", str(num_designs), "--num_timesteps", str(steps),
            "--seed", str(RFD3_SEED)]


def _rfd3_atom_digest(cif: Path) -> str:
    """sha256 of the coordinate records only, so a CIF-writer metadata or path field
    cannot make two identical structures look different (or two different ones look
    the same)."""
    lines = [ln for ln in cif.read_text().splitlines()
             if ln.startswith(("ATOM", "HETATM"))]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def run_rfd3(keep: bool) -> dict:
    """Design, parse, and structurally score RFD3. Returns a result row."""
    out = REPO_ROOT / "rfd3_gate_designs"
    if out.exists():
        shutil.rmtree(out)  # never score a stale run if this design run crashes

    print(f"\n{'='*70}\n[rfd3] designing {RFD3_SPEC.name}:{RFD3_SPEC_ID} "
          f"({RFD3_NUM_DESIGNS} designs, {RFD3_TIMESTEPS} steps, seed {RFD3_SEED})"
          f"\n{'='*70}", flush=True)

    row = {"model": "rfd3", "seconds": None, "n_designs": 0, "in_band": None,
           "breaks": None, "clash_frac": None, "clashes": None, "distinct_aa": None,
           "unk": None, "clean_rate": None, "determinism": None, "parse": False,
           "gate": False, "error": None, "per_design": []}

    try:
        designed = _rfd3_designed_residues()
        geom = _load_geometry_harness()
    except Exception as e:
        row["error"] = f"host featurize/harness failed: {type(e).__name__}: {e}"
        return row
    print(f"[rfd3] designed residues from the host featurizer: {len(designed)}", flush=True)

    t0 = time.monotonic()
    rc, timed_out = _run_fold(_rfd3_design_cmd(out, RFD3_NUM_DESIGNS, RFD3_TIMESTEPS),
                              FOLD_TIMEOUT_S, cwd=REPO_ROOT)
    row["seconds"] = time.monotonic() - t0
    if timed_out:
        row["error"] = f"design timed out after {FOLD_TIMEOUT_S}s"
        return row
    if rc != 0:
        row["error"] = f"design exited {rc}"
        return row

    cifs = sorted(out.rglob("*.cif"))
    row["n_designs"] = len(cifs)
    # Assert the leg ran the model it names before reading any margin: a scoring pass over
    # zero or one design would otherwise report a clean number for a run that never happened.
    if len(cifs) != RFD3_NUM_DESIGNS:
        row["error"] = f"expected {RFD3_NUM_DESIGNS} designs, found {len(cifs)} CIF(s)"
        return row
    try:
        _parse_gate(cifs, name="rfd3")
        row["parse"] = True
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row

    try:
        row["per_design"] = [_rfd3_score_cif(c, designed, geom) for c in cifs]
    except Exception as e:
        row["error"] = f"scoring failed: {type(e).__name__}: {e}"
        return row
    for cif, m in zip(cifs, row["per_design"]):
        print(f"[rfd3] {cif.name}: designed {m['n_designed']}, CA-CA median "
              f"{m['step_median']} A, in band {m['in_band']:.4f}, breaks {m['breaks']}, "
              f"clashes {m['clashes']}/{m['heavy']} ({m['clash_frac']:.6f}, worst "
              f"{m['worst_contact']} A), distinct AA {m['distinct_aa']}, UNK {m['unk']}, "
              f"top AA {m['top_aa_frac']:.1%}", flush=True)

    # Geometry is a clean RATE (see the floors above: an occasional broken backbone is real
    # RFD3 behaviour, so requiring every design to be clean would fail correct code). Every
    # other metric aggregates on the WORST design, since a mean over four hides one bad one.
    for m in row["per_design"]:
        m["clean"] = m["breaks"] <= RFD3_MAX_BREAKS and m["in_band"] >= RFD3_MIN_INBAND
    row["clean_rate"] = (sum(m["clean"] for m in row["per_design"])
                         / len(row["per_design"]))
    row["in_band"] = min(m["in_band"] for m in row["per_design"])
    row["breaks"] = max(m["breaks"] for m in row["per_design"])
    row["clash_frac"] = max(m["clash_frac"] for m in row["per_design"])
    row["clashes"] = max(m["clashes"] for m in row["per_design"])
    row["distinct_aa"] = min(m["distinct_aa"] for m in row["per_design"])
    row["unk"] = max(m["unk"] for m in row["per_design"])

    # Determinism: the same seed in two fresh processes must write the same coordinates.
    # This is the arm that sees the tile-sparsity / unmasked-tail class, which every
    # single-run geometry number is blind to.
    digests = []
    for rep in (1, 2):
        rep_out = REPO_ROOT / f"rfd3_gate_determinism_{rep}"
        shutil.rmtree(rep_out, ignore_errors=True)
        rc, timed_out = _run_fold(_rfd3_design_cmd(rep_out, 1, RFD3_DET_TIMESTEPS),
                                  FOLD_TIMEOUT_S, cwd=REPO_ROOT)
        reps = sorted(rep_out.rglob("*.cif"))
        if timed_out or rc != 0 or len(reps) != 1:
            row["error"] = (f"determinism repeat {rep} failed (rc={rc}, "
                            f"timed_out={timed_out}, {len(reps)} CIF)")
            return row
        digests.append(_rfd3_atom_digest(reps[0]))
        if not keep:
            shutil.rmtree(rep_out, ignore_errors=True)
    row["determinism"] = digests[0] == digests[1]
    print(f"[rfd3] determinism ({RFD3_DET_TIMESTEPS} steps, seed {RFD3_SEED}, two fresh "
          f"processes): {digests[0][:16]} vs {digests[1][:16]} -> "
          f"{'identical' if row['determinism'] else 'DIFFER'}", flush=True)

    row["gate"] = (row["clean_rate"] >= RFD3_MIN_CLEAN_RATE
                   and row["clashes"] <= RFD3_MAX_CLASHES
                   and row["distinct_aa"] >= RFD3_MIN_DISTINCT_AA
                   and row["unk"] <= RFD3_MAX_UNK
                   and row["determinism"])

    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def run_pxdesign(keep: bool) -> dict:
    """Design one binder through the shipped CLI path, parse it, and score the conditioning."""
    import hashlib

    import torch

    out = REPO_ROOT / "pxdesign_gate_designs"
    if out.exists():
        shutil.rmtree(out)  # never score a stale run if this design run crashes

    print(f"\n{'='*70}\n[pxdesign] designing against {PXDESIGN_SPEC.name} "
          f"({PXDESIGN_NUM_DESIGNS} design, {PXDESIGN_N_STEP} steps)\n{'='*70}", flush=True)

    row = {"model": "pxdesign", "seconds": None, "fit_rmsd": None, "sha16": None,
           "binder_residues": None, "parse": False, "gate": False, "error": None}
    try:
        from tt_bio.pxdesign.inputs import design_inputs_from_yaml
        from tt_bio.pxdesign.model import ProtenixDesign
        from tt_bio.pxdesign.write import write_design_cifs
        from tt_bio.main import ensure_pxdesign_weights, ensure_p300_mesh_descriptor
    except Exception as e:
        row["error"] = f"import failed: {type(e).__name__}: {e}"
        return row

    t0 = time.monotonic()
    try:
        feats = design_inputs_from_yaml(PXDESIGN_SPEC)
        feats = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float64 else v)
                 for k, v in feats.items()}
        ckpt = ensure_pxdesign_weights(Path(os.path.expanduser("~/.boltz")))
        ensure_p300_mesh_descriptor()
        model = ProtenixDesign.load_from_checkpoint(str(ckpt))
        coords = model.design(feats, n_step=PXDESIGN_N_STEP, n_sample=PXDESIGN_NUM_DESIGNS,
                              seed=PXDESIGN_SEED)
        rows = write_design_cifs(coords, feats, out, stem="PDL1")
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    row["seconds"] = time.monotonic() - t0
    row["fit_rmsd"] = max(r["fit_rmsd"] for r in rows)
    row["binder_residues"] = rows[0]["binder_residues"]
    row["sha16"] = hashlib.sha256(coords.contiguous().numpy().tobytes()).hexdigest()[:16]

    try:
        _parse_gate(sorted(out.rglob("*.cif")), name="pxdesign")
        row["parse"] = True
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row

    row["gate"] = row["fit_rmsd"] <= PXDESIGN_MAX_FIT_RMSD
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
        dproc = subprocess.Popen(
            [OPENDDE_DOCKQ_PYTHON, str(dockq_script), str(conf_cif),
             str(OPENDDE_ABAG_NATIVE), "--out", str(out_json)],
            cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            start_new_session=True)
        dout, derr = dproc.communicate(timeout=DOCKQ_TIMEOUT_S)
        if dproc.returncode != 0:
            row["error"] = (f"DockQ exited {dproc.returncode}: "
                            f"{(derr or dout).strip()[:200]}")
            return row
        import json
        with open(out_json) as fp:
            dq = json.load(fp)
        row["dockq"] = float(dq["global_dockq"])
        # mean fnat over native interfaces, for visibility (the paratope-epitope face)
        fnats = [v.get("fnat") for v in dq["interfaces"].values()
                 if v.get("fnat") is not None]
        row["fnat"] = (sum(fnats) / len(fnats)) if fnats else None
    except subprocess.TimeoutExpired:
        # Named like the fold-timeout legs, and ahead of the generic handler so a hang reads
        # as a timeout rather than an opaque "DockQ eval failed".
        _kill_group(dproc)
        row["error"] = (f"DockQ timed out after {DOCKQ_TIMEOUT_S}s "
                        f"(raise RELEASE_GATE_DOCKQ_TIMEOUT on a slow host)")
        return row
    except Exception as e:
        row["error"] = f"DockQ eval failed: {e}"
        return row

    row["gate"] = row["dockq"] >= OPENDDE_ABAG_MIN_DOCKQ

    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def _fold_error(text: str) -> str:
    """A failed fold's OWN error line, not whatever happened to print last.

    Tailing the log is wrong whenever the wrapper outlives the fold:
    `lever_census.py` prints its lever table AFTER the CLI it wraps exits, so the
    last three lines of a crashed census fold are the table. That is how four
    size-ladder models reported one identical, meaningless string
    ("census fold exited 1: B2_TOKEN_DIT_SDPA False served=None ...") for a
    TypeError, and why the real cause needed a fresh root-cause pass instead of
    one read. Prefer, in order: tt_bio.main's own "✗ <job>: <msg>" failure line,
    the exception line of the last traceback, then the tail as a last resort.
    """
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""
    fails = [ln.strip() for ln in lines if ln.lstrip().startswith("✗ ")]
    if fails:
        return fails[-1][:400]
    tb = [i for i, ln in enumerate(lines)
          if ln.startswith("Traceback (most recent call last)")]
    if tb:
        # The frames are indented; the first unindented "Name: message" after them
        # is the exception.
        for ln in lines[tb[-1] + 1:]:
            if not ln[:1].isspace() and ": " in ln:
                return ln[:400]
    # No explicit failure line. Prefer any line that at least mentions a fault over a
    # blind tail; origin/main solved the same problem that way and it beats the tail
    # whenever the wrapper outlives the fold.
    marks = ("\u2717", "Traceback", "Error", "FATAL", "failed:")
    hits = [ln for ln in lines if any(m in ln for m in marks)]
    return " / ".join((hits or lines)[-3:])[:400]


def run_nesso1(keep: bool) -> dict:
    """Score Nesso-1's eleven output scalars against the torch reference. Returns a row.

    Runs the parity harness as a subprocess rather than importing it: it opens a device
    context, and this process must stay free to run the other arms after it.
    """
    if not NESSO1_FIXTURE.exists():
        return {"model": "nesso1", "seconds": None, "x_over_r": None, "spread": None,
                "n_tokens": None, "gate": False,
                "error": f"missing nesso1 parity fixture {NESSO1_FIXTURE}"}

    # perf/nesso1/, not the repo root: measurement artifacts belong under perf/ (the 08-13
    # run_*.sh lesson), and --keep leaves this file behind on purpose.
    out_json = REPO_ROOT / "perf" / "nesso1" / "gate_parity.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(NESSO1_PARITY),
        "--fixture", str(NESSO1_FIXTURE),
        "--trunk", NESSO1_TRUNK, "--affinity", NESSO1_AFFINITY,
        "--repeats", str(NESSO1_REPEATS),
        "--json", str(out_json),
    ]
    print(f"\n{'='*70}\n[nesso1] scoring {NESSO1_FIXTURE.name} against the torch reference "
          f"({NESSO1_TRUNK} trunk, {NESSO1_AFFINITY} affinity, {NESSO1_REPEATS} repeats)"
          f"\n{'='*70}", flush=True)

    row = {"model": "nesso1", "seconds": None, "x_over_r": None, "spread": None,
           "n_tokens": None, "gate": False, "error": None}
    t0 = time.monotonic()
    rc, timed_out = _run_fold(cmd, NESSO1_TIMEOUT_S, cwd=REPO_ROOT)
    row["seconds"] = time.monotonic() - t0
    if timed_out:
        row["error"] = f"device_parity timed out after {NESSO1_TIMEOUT_S}s"
        return row
    # rc is 1 on a FAIL verdict, which is a result rather than a crash, so read the JSON
    # first and only call a missing report an error.
    if not out_json.exists():
        row["error"] = f"device_parity exited {rc} and wrote no report"
        return row
    rep = json.loads(out_json.read_text())
    if not keep:
        out_json.unlink()
    row["n_tokens"] = rep["n_tokens"]
    row["x_over_r"] = rep["X_over_R"]
    row["spread"] = rep["max_device_spread"]
    row["floors"] = rep["floors"]
    row["worst_key"] = rep["X_device_vs_torch_key"]
    row["gate"] = rep["verdict"] == "PASS"
    if not row["gate"]:
        row["error"] = (f"worst scalar {rep['X_device_vs_torch_key']} at "
                        f"{rep['X_over_R']:.3f}xR, device spread {rep['max_device_spread']:.3g}")
    return row


def run_rf3_1024aa(keep: bool) -> dict:
    """Fold the 997 aa RF3 anchor on the device and gate its CA-RMSD to the crystal.

    Delegates to ``scripts/rf3_port/accuracy_cell.py`` as a subprocess, the same way the
    nesso1 leg delegates to its own parity harness: the cell opens a device context and
    this process must stay free for the arms after it.
    """
    row = {"model": "rf3-1024aa", "seconds": None, "xtal_a": None, "ref_xtal_a": None,
           "x_a": None, "n_ca": None, "gate": False, "error": None}
    ref_seed = RF3_1024AA_REF_CACHE / f"seed{RF3_1024AA_SEED}.npz"
    if not ref_seed.exists():
        # Refuse rather than let the cell fall through to computing one: a 997 aa reference
        # trunk plus rollout on the host is hours, not the minutes this leg is budgeted at.
        row["error"] = f"missing committed reference cache {ref_seed}"
        return row

    # perf/rf3/, not the repo root: measurement artifacts belong under perf/, and --keep
    # leaves the device coordinates and the report behind on purpose.
    work = REPO_ROOT / "perf" / "rf3" / "gate_1024aa"
    if work.exists():
        shutil.rmtree(work)  # a cached rollout here would score a previous tree
    work.mkdir(parents=True, exist_ok=True)
    out_json = work / "report.json"
    cmd = [
        sys.executable, str(RF3_1024AA_CELL),
        "--fixture", RF3_1024AA_FIXTURE,
        "--seeds", str(RF3_1024AA_SEED),
        "--steps", str(_sampling_steps("rf3")),
        "--work", str(work),
        "--ref-cache", str(RF3_1024AA_REF_CACHE),
        "--out", str(out_json),
    ]
    print(f"\n{'='*70}\n[rf3-1024aa] folding {RF3_1024AA_FIXTURE} on the device and "
          f"scoring it against the crystal (seed {RF3_1024AA_SEED}, "
          f"{_sampling_steps('rf3')} steps, floor <={RF3_1024AA_MAX_XTAL_A} A)"
          f"\n{'='*70}", flush=True)

    t0 = time.monotonic()
    rc, timed_out = _run_fold(cmd, RF3_1024AA_TIMEOUT_S, cwd=REPO_ROOT)
    row["seconds"] = time.monotonic() - t0
    if timed_out:
        row["error"] = f"accuracy_cell timed out after {RF3_1024AA_TIMEOUT_S}s"
        return row
    if rc != 0:
        row["error"] = f"accuracy_cell exited {rc}"
        return row
    if not out_json.exists():
        row["error"] = f"accuracy_cell exited {rc} and wrote no report"
        return row
    rep = json.loads(out_json.read_text())
    vs = rep.get("vs_crystal")
    if not vs:
        row["error"] = ("report carries no vs_crystal block; "
                        f"is ground_truth_ca.json still next to the {RF3_1024AA_FIXTURE} "
                        "fixture?")
        return row
    row["n_ca"] = vs["n_ca_compared"]
    # The worst seed of however many ran, so lengthening the seed list can only tighten
    # this leg. At one seed it is that seed.
    row["xtal_a"] = max(r["device_vs_xtal_A"] for r in vs["per_seed"])
    row["ref_xtal_a"] = max(r["reference_vs_xtal_A"] for r in vs["per_seed"])
    # X is reported, not gated -- see the constants for why the crystal carries the floor.
    ca = (rep.get("metrics") or {}).get("kabsch_rmsd") or {}
    row["x_a"] = ca.get("cross", {}).get("mean")
    row["gate"] = row["xtal_a"] <= RF3_1024AA_MAX_XTAL_A
    if not row["gate"]:
        row["error"] = (f"device sits {row['xtal_a']:.3f} A from the crystal, floor "
                        f"<={RF3_1024AA_MAX_XTAL_A} A")
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
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
        row["error"] = f"predict exited {rc}: {_fold_error(text)}"
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


def _size_ladder_precondition(model: str):
    """A one-line reason this model cannot be folded here, or None.

    Checked once before the first fold rather than discovered by one. nesso1 needs the
    checkpoint's uncommitted 413 MB ccd.pkl, and without it every rung fails with a
    FileNotFoundError from inside a subprocess — twelve wasted model loads and an arm that
    reads as broken instead of unconfigured. An arm that fails for a reason nobody can act on
    is an arm someone switches off.

    It DOWNLOADS on a miss rather than refusing. find_ccd fetches the file now, so refusing
    here would skip an arm that would have run: the first rung would have downloaded it
    anyway. Doing it in the precondition just moves the one-time 413 MB out of the fold loop,
    which is what "before any device work" was always for. What still fails by name is the
    case nobody can fix by waiting: no file on disk and no way to fetch one.
    """
    if model != "nesso1":
        return None
    try:
        from tt_bio.nesso1_input import find_ccd
        find_ccd(os.environ.get("NESSO_CACHE"))
    except Exception as e:                                               # noqa: BLE001
        return f"nesso1 precondition: {e}"
    return None


def _size_ladder_fixture(model: str, rung: int) -> Path:
    """The rung's input. nesso1 needs a ligand and an affinity property, so it brings its own
    ladder; every other model folds the shared apo fixture."""
    if model == "nesso1":
        return (REPO_ROOT / "perf" / "nesso1" / "inputs" / "ladder" / f"aa{rung}"
                / f"cdk2_{rung}.yaml")
    return REPO_ROOT / "perf" / "size512" / "fixtures" / f"cdk2x2_{rung}.yaml"


def _affinity_seconds(out_dir: Path):
    """The forward's own seconds from affinity.csv: nesso1's runtime_s."""
    csv_path = out_dir / "affinity.csv"
    if not csv_path.exists():
        return None
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))
    vals = [float(r["seconds"]) for r in rows if r.get("seconds") and not r.get("error")]
    return max(vals) if vals else None


def lever_census_flags() -> tuple:
    """The census's own lever names, so --size-ladder-record-lever cannot be given a typo."""
    path = REPO_ROOT / "scripts" / "lever_census.py"
    spec = importlib.util.spec_from_file_location("tt_bio_lever_census", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return tuple(f for f, *_rest in mod.LEVERS)


def _run_census_fold(model: str, rung: int, workdir: Path, tag: str,
                     need_runtime: bool = True) -> dict:
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
    fixture = _size_ladder_fixture(model, rung)
    if not fixture.exists():
        return {"error": f"missing size-ladder fixture {fixture}"}
    label = f"{model}-{rung}-{tag}"
    census_json = workdir / f"census_{label}.json"
    out_dir = workdir / f"out_{label}"
    log = workdir / f"{label}.log"
    shutil.rmtree(out_dir, ignore_errors=True)
    workdir.mkdir(parents=True, exist_ok=True)
    census = [sys.executable, str(REPO_ROOT / "scripts" / "lever_census.py"),
              "--tt-bio", sys.executable, "--label", label, "--out", str(census_json), "--"]
    if model == "nesso1":
        # `tt-bio affinity`, not `predict` — see the SIZE_LADDER_NESSO_* block. The census hook
        # runs in every process the CLI starts, launcher included, and this model folds in the
        # launcher, so the counters land the same way they do for a spawned predict worker.
        cmd = census + [
            "-m", "tt_bio.main", "affinity", str(fixture),
            "--model", "nesso1", "--trunk", "bf16",
            "--recycling_steps", str(SIZE_LADDER_NESSO_RECYCLING),
            "--tokens_budget", str(SIZE_LADDER_NESSO_TOKENS_BUDGET),
            "--out_dir", str(out_dir),
        ]
        if os.environ.get("NESSO_CACHE"):
            cmd += ["--cache", os.environ["NESSO_CACHE"]]
    else:
        cmd = census + [
            "-m", "tt_bio.main", "predict", str(fixture),
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
        return {"error": f"census fold exited {rc}: "
                         f"{_fold_error(log.read_text(errors='replace'))}"}
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
        # The clause the guard declined on, aggregated by reason (see
        # lever_census.REJECTS_ATTR). This is what makes an exemption entry evidence:
        # "dark because m_tiles=64 does not divide the tuned block" instead of a story.
        if r.get("rejects"):
            levers[r["flag"]]["rejects"] = r["rejects"]
    if model == "nesso1":
        runtime_s = _affinity_seconds(out_dir)
        where = "affinity.csv"
    else:
        results = out_dir / predict_results_dir_name(model, fixture.stem) / "results.json"
        where = results.name
        runtime_s = None
        if results.exists():
            try:
                rows = json.loads(results.read_text())
                ts = [row["runtime_s"] for row in rows
                      if row.get("status") == "ok" and row.get("runtime_s") is not None]
                runtime_s = max(ts) if ts else None
            except Exception:
                runtime_s = None
    if runtime_s is None and need_runtime:
        return {"error": f"no runtime_s in {where} (fold ok but timing missing)"}
    return {"levers": levers, "runtime_s": runtime_s, "wall": wall,
            "census_json": census_json, "grid": census.get("grid")}


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


def _size_ladder_clause_finding(b: dict, c: dict, flag: str, where: str):
    """Same fired fraction, different clause: the guard is refusing for a reason it did not
    refuse for when the baseline was taken. That is a behaviour change with no timing
    signature at all, so nothing else in this arm can see it — but two states are not that,
    and reading them as one is how this arm went red on main for an instrument change.

    A missing `rejects` means the clause was NOT MEASURED, not "declined for no reason":

      * The baseline has no clause but had declines to record one for. Every baseline taken
        before 95033b2f is in that state for the six wrap-counted levers, because the census
        could not report their clause yet. Compared against today's census that reads as three
        guards changing their mind on every model at every rung, and it is why the arm has been
        failing on main since that commit rather than since anything changed behaviour.
      * Neither side declined. A guard with 0 declines has no clause to have; a clause sitting
        on such an entry is a recording artifact, which is what REBLOCK_PERMUTE_GATED carried
        from the shared REJECTS dict it used to be misattributed.

    THE RULE this implies, alongside the L1-budget leg's: an instrument change that widens what
    the baseline compares re-records in the same commit. Adding a field to the census and not
    re-recording turns the arm red for a reason no reviewer can distinguish from a real one.
    """
    if not b.get("rejects") and (b.get("declined") or 0) > 0:
        return None
    if not (b.get("declined") or 0) and not (c.get("declined") or 0):
        return None
    bs, cs = sorted(b.get("rejects") or {}), sorted(c.get("rejects") or {})
    return None if bs == cs else f"{where} {flag}: decline clause {bs} -> {cs}"


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
            # Name the clause it went dark ON: that is the mechanism, and it is the
            # difference between "K2 stopped firing" and "K2 stopped firing because
            # fill_preconditions rejects a padded mask", which is the actual defect.
            clause = ", ".join(sorted(c.get("rejects") or {})) if fc == 0.0 else ""
            findings.append(f"{where} {flag}: frac {fb:.3f} -> {fc:.3f} "
                            f"({'went dark' if fc == 0.0 else 'started firing'}"
                            + (f" on {clause}" if clause else "") + ")")
        elif _size_ladder_clause_finding(b, c, flag, where):
            findings.append(_size_ladder_clause_finding(b, c, flag, where))
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
    """Census-fold every rung, discarding the first fold AT EACH RUNG, then report.

    Returns {"levers": {rung: ...}, "runtime_s": {rung: median}, "sigma": relative
    runtime noise at 512 | None, "census_jsons": {rung: path}} or {"error": ...}.

    The discard is per rung, not one warm-up at the smallest rung, because the JIT
    kernel cache is keyed by SHAPE: folding 256 aa does not warm 512 aa. Measured on
    esmfold2 at 512 aa, five reps after a 256 aa warm-up: 82.3, 47.9, 49.8, 51.0,
    52.5 s. Keeping the first reads sigma = 25.4 % and a +-2.66 exponent band, which
    would have recorded esmfold2 as "cannot be gated cheaply"; dropping it reads
    sigma = 3.9 % and +-0.40, tighter than boltz-2. The old one-warm-up-per-model
    policy was calibrated on boltz-2, where a single fold is enough -- the same
    one-size-fits-all mistake this whole arm exists to catch, in the arm itself.
    """
    levers, runtimes, census_jsons = {}, {}, {}
    sigma, grid = None, None
    for rung in rungs:
        reps = reps_512 if rung == 512 else reps_other
        runs = []
        for rep in range(reps + 1):
            r = _run_census_fold(model, rung, workdir,
                                 "warmup" if rep == 0 else f"rep{rep - 1}")
            if r.get("error"):
                return {"error": f"rung {rung} "
                                 f"{'warm-up' if rep == 0 else f'rep {rep - 1}'}: "
                                 f"{r['error']}"}
            if rep == 0:
                continue          # cold: kernels for this shape compile on this fold
            grid = grid or r.get("grid")
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
            "census_jsons": census_jsons, "grid": grid}


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
                clause = ", ".join(f"{k} x{v}" for k, v in
                                   sorted((e.get("rejects") or {}).items(),
                                          key=lambda kv: -kv[1])[:3])
                e["reason"] = ("TODO: say why this is legitimate at this size"
                               + (f" (declines on {clause})" if clause else ""))
                todo += 1
    return todo


def _size_ladder_check_model(model: str, rungs, base_model: dict, workdir: Path) -> dict:
    pre = _size_ladder_precondition(model)
    if pre:
        return {"model": model, "gate": False, "error": pre, "findings": [pre]}
    reps = base_model.get("reps", 1)
    meas = _size_ladder_measure_model(model, rungs, workdir, reps, reps)
    if meas.get("error"):
        return {"model": model, "gate": False, "error": meas["error"],
                "findings": [meas["error"]]}
    findings = []
    b_grid, c_grid = base_model.get("grid"), meas.get("grid")
    if b_grid and c_grid and b_grid != c_grid:
        # Not a warning. A guard sized against the core grid flips with it (protenix-v2's K2 is
        # admitted on 11x10 and refused on 13x10), so a census compared across grids reports
        # levers going dark that never went dark. Board type alone does not pin the grid:
        # harvesting means one board type presents several.
        return {"model": model, "gate": False,
                "error": f"baseline recorded on a {b_grid} grid, this card presents {c_grid} — "
                         f"lever verdicts are grid-dependent, so re-record on this grid "
                         f"(--size-ladder-record) rather than comparing across them",
                "findings": [f"{model}: grid {b_grid} -> {c_grid}"],
                "runtime_s": meas["runtime_s"], "exponents": {}}
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
            "runtime_s": meas["runtime_s"], "exponents": measured_k,
            "baseline_from": " ".join(str(base_model.get(k)) for k in
                                      ("recorded", "host", "commit") if base_model.get(k))}


def _size_ladder_lever_todo(entry: dict) -> str:
    """The TODO a dark lever gets when it has no exemption reason yet, carrying the clause it
    declined on so filling it in is confirming a measurement rather than writing a story."""
    clause = ", ".join(f"{k} x{v}" for k, v in
                       sorted((entry.get("rejects") or {}).items(), key=lambda kv: -kv[1])[:3])
    return ("TODO: say why this is legitimate at this size"
            + (f" (declines on {clause})" if clause else ""))


def run_size_ladder_add_lever(flags, keep: bool, baseline_path: Path,
                              models=None) -> dict:
    """Add census levers to an existing baseline without re-measuring its timings.

    A counter-only lever — a guard that already shipped, given a `*_STATS` pair so the census
    can finally see it — makes check mode report "new lever not in the baseline" for every
    model at every rung. The only recorded fix is a full re-record: 60 folds, hours of device
    time, and it throws away a timing baseline measured on a quiet host to replace it with one
    measured on whatever host was free. That is a lot of evidence discarded to admit a counter
    that changes no behaviour, and it will happen again every time someone instruments a guard.

    So: fold each (model, rung) ONCE and splice in only the named levers' rows. What makes that
    honest rather than convenient is the refusal — every OTHER lever in the census must still
    match the baseline exactly, by the same comparator check mode uses, or the splice is
    refused for that model and the message says to do a full re-record. Nothing can be
    laundered through here. It doubles as a free check-mode run of the arm's census half, and
    as the proof that a cold fold's guard decisions equal a warm one's (the recorded baseline
    discards the first fold at each rung; this mode does not, and the comparison is exact).

    Because the comparison passed, every other lever's counts are identical, so this also writes
    back the decline clauses the census can now measure and the baseline never recorded — see
    `_size_ladder_clause_finding` for why an absent clause is "not measured" rather than "none".
    That is the same act as adding the lever: recording a measurement the file was missing.

    Each spliced row is stamped in `levers_added`, so the entry says which of its numbers came
    from a different host at a different commit than the rest.
    """
    models = list(models or SIZE_LADDER_MODELS)
    flags = [f for f in (flags.split(",") if isinstance(flags, str) else flags) if f]
    rungs = SIZE_LADDER_RUNGS
    card = _size_ladder_card_type()
    workdir = SIZE_LADDER_WORKDIR
    unknown = [f for f in flags if f not in lever_census_flags()]
    if unknown:
        return {"model": "size-ladder", "seconds": 0, "gate": False, "card": card,
                "error": f"not levers in scripts/lever_census.py: {', '.join(unknown)}",
                "legs": []}
    try:
        baseline = json.loads(baseline_path.read_text())
    except Exception as e:
        return {"model": "size-ladder", "seconds": 0, "gate": False, "card": card,
                "error": f"baseline {baseline_path} unreadable: {e}", "legs": []}
    card_block = baseline.get("cards", {}).get(card)
    if card_block is None:
        return {"model": "size-ladder", "seconds": 0, "gate": False, "card": card,
                "error": f"NO BASELINE for card type '{card}' — there is nothing to add a "
                         f"lever to; record one with --size-ladder-record", "legs": []}
    print(f"\n{'='*70}\n[size-ladder] adding {', '.join(flags)} to the {card} baseline: "
          f"{', '.join(models)} at rungs {','.join(map(str, rungs))}, one fold per rung"
          f"\n{'='*70}", flush=True)
    t0 = time.monotonic()
    stamp = f"{time.strftime('%Y-%m-%d')} {socket.gethostname()} {_repo_commit()}"
    legs = []
    for m in models:
        base_model = card_block.get("models", {}).get(m)
        if base_model is None:
            err = f"{m}: not in the {card} baseline — record it first"
            legs.append({"model": m, "gate": False, "error": err, "findings": [err]})
            continue
        pre = _size_ladder_precondition(m)
        if pre:
            legs.append({"model": m, "gate": False, "error": pre, "findings": [pre]})
            continue
        measured, clauses, findings, grid = {}, {}, [], None
        for rung in rungs:
            # need_runtime=False: this mode compares census counts and writes one lever's
            # row. It never reads a timing, so a fold whose results.json is not readable the
            # instant the subprocess exits must not refuse the splice — openfold3 writes its
            # results through a lock/.bak rename and lost that race twice in a row here, which
            # read as "openfold3 refused" and hid the reason behind a missing report line.
            r = _run_census_fold(m, rung, workdir, "addlever", need_runtime=False)
            if r.get("error"):
                findings.append(f"{m}/{rung}: {r['error']}")
                break
            grid = grid or r.get("grid")
            base_levers = base_model.get("levers", {}).get(str(rung))
            if base_levers is None:
                findings.append(f"{m}/{rung}: rung not recorded in the baseline")
                continue
            already = [f for f in flags if f in base_levers]
            if already:
                findings.append(f"{m}/{rung}: already in the baseline: {', '.join(already)}")
                continue
            where = f"{m}/{rung}"
            expected = {f"{where} {f}: new lever not in the baseline "
                        f"(re-record with --size-ladder-record)" for f in flags}
            other = [f for f in _size_ladder_compare_levers(base_levers, r["levers"], where)
                     if f not in expected]
            if other:
                findings.extend(other)
                continue
            measured[str(rung)] = {f: r["levers"][f] for f in flags}
            # The comparison above passed, so every other lever's counts and clause are either
            # identical or in one of the two not-measured states. Writing the measured clause
            # back is therefore recording what is already true, and it is the only way an old
            # baseline stops carrying an unenforceable clause field forever.
            clauses[str(rung)] = {f: e.get("rejects") for f, e in r["levers"].items()
                                  if f not in flags and f in base_levers
                                  and (e.get("rejects") or None)
                                      != (base_levers[f].get("rejects") or None)}
        b_grid = base_model.get("grid")
        if grid and b_grid and grid != b_grid:
            findings.append(f"{m}: baseline recorded on a {b_grid} grid, this card presents "
                            f"{grid} — lever verdicts are grid-dependent, re-record instead")
        if findings or len(measured) != len(rungs):
            legs.append({"model": m, "gate": False, "findings": findings,
                         "error": "; ".join(findings) or
                                  f"{m}: only {len(measured)}/{len(rungs)} rungs measured"})
            continue
        n_clauses = 0
        for rung, entries in measured.items():
            for f, entry in entries.items():
                if _size_ladder_dark(entry):
                    entry["reason"] = _size_ladder_lever_todo(entry)
                base_model["levers"][rung][f] = entry
            for f, rej in clauses.get(rung, {}).items():
                base_model["levers"][rung][f]["rejects"] = rej
                n_clauses += 1
        for f in flags:
            base_model.setdefault("levers_added", {})[f] = stamp
        baseline_path.write_text(json.dumps(baseline, indent=2) + "\n")
        legs.append({"model": m, "gate": True, "error": None, "findings": [],
                     "added": {rung: {f: (e["served"], e["declined"])
                                      for f, e in es.items()}
                               for rung, es in measured.items()}})
        print(f"[size-ladder] {m}: {', '.join(flags)} added at {len(measured)} rungs, "
              f"every other lever "
              f"unchanged" + (f", {n_clauses} clause field(s) recorded" if n_clauses else ""),
              flush=True)
    gate = bool(legs) and all(l["gate"] for l in legs)
    todos = sum(1 for m in models for rung in rungs for f in flags
                for e in [((card_block.get("models", {}).get(m) or {})
                           .get("levers", {}).get(str(rung), {}).get(f))]
                if e and str(e.get("reason", "")).startswith("TODO"))
    if todos:
        print(f"[size-ladder] {todos} rung(s) need a one-line exemption reason — "
              f"search TODO in {baseline_path}; the check FAILS without one.", flush=True)
    if not keep:
        shutil.rmtree(workdir, ignore_errors=True)
    return {"model": "size-ladder", "seconds": time.monotonic() - t0, "gate": gate,
            "card": card, "error": next((l["error"] for l in legs if l["error"]), None),
            "legs": legs}


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
          f"{','.join(map(str, rungs))}\n[size-ladder] predict folds: "
          f"{SIZE_LADDER_STEPS} steps, 1 sample, seed {SEED}, single-sequence; "
          f"nesso1: tt-bio affinity, bf16 trunk, {SIZE_LADDER_NESSO_RECYCLING} recycles, "
          f"{SIZE_LADDER_NESSO_TOKENS_BUDGET}-token crop\n{'='*70}", flush=True)
    t0 = time.monotonic()
    legs = []
    if record:
        card_block = baseline.get("cards", {}).get(card, {})
        old_models = card_block.get("models", {})
        # Seeded with the card's existing models, not empty: recording a subset
        # (--size-ladder-models) then UPDATES those models and leaves the rest of
        # the card block intact. A 6-model record is ~2 h of device time, so it
        # has to be resumable a model at a time instead of all-or-nothing.
        stamp = {"recorded": time.strftime("%Y-%m-%d"), "host": socket.gethostname(),
                 "commit": _repo_commit()}
        # The card-level stamp describes the LAST record pass, so on a subset record
        # (--size-ladder-models) it stops describing the models that pass did not touch.
        # rf3 was recorded on qb1 while the other five were recorded on pc; without a
        # per-model stamp the file then claims all six came from qb1. So every entry
        # carries its own, and an entry from before this existed inherits the card-level
        # stamp it WAS recorded under, which is the one being overwritten here.
        old_stamp = {k: baseline.get("cards", {}).get(card, {}).get(k)
                     for k in ("recorded", "host", "commit")}
        carried = {}
        for m_old, e_old in old_models.items():
            e_old = dict(e_old)
            for k, v in old_stamp.items():
                if v is not None:
                    e_old.setdefault(k, v)
            carried[m_old] = e_old
        new_card = {**stamp, "models": carried}
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
            pre = _size_ladder_precondition(m)
            if pre:
                legs.append({"model": m, "gate": False, "error": pre, "findings": [pre]})
                continue
            meas = _size_ladder_measure_model(m, rungs, workdir,
                                              SIZE_LADDER_SIGMA_REPS, 1)
            if meas.get("error"):
                legs.append({"model": m, "gate": False, "error": meas["error"],
                             "findings": [meas["error"]]})
                continue
            block, skip = _size_ladder_exponent_block(meas["runtime_s"], meas["sigma"])
            todos += _size_ladder_fill_reasons(meas["levers"],
                                               old_models.get(m, {}).get("levers"))
            entry = {"grid": meas.get("grid"), **stamp,
                     "runtime_s": meas["runtime_s"], "levers": meas["levers"]}
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
            # Beside the baseline being written, not always the committed directory:
            # --size-ladder-baseline exists so a smoke run can record to scratch, and a
            # fixed provenance path made that scratch run overwrite the committed
            # evidence for the REAL baseline (hit 2026-08-22 by a 256-rung smoke).
            prov = (SIZE_LADDER_PROVENANCE
                    if baseline_path.resolve() == SIZE_LADDER_BASELINE.resolve()
                    else baseline_path.parent / f"{baseline_path.stem}_census")
            prov.mkdir(parents=True, exist_ok=True)
            for rung, cj in meas["census_jsons"].items():
                shutil.copy(cj, prov / f"census_{m}_{rung}_{card}.json")
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
        for part, grid, l1, dram, _prov in L1_BUDGET_PARTS:
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
            # Host-concat budget, the DRAM-keyed figure. Asserted as behaviour, not as its own
            # arithmetic: a <=12 GiB part must keep the measured 1.5 GiB exactly, a >=16 GiB part
            # must not still be running the 12 GiB figure, and no part may tighten.
            budget = tt._concat_host_budget(dram)
            row["checks"] += 1
            if budget < tt.CONCAT_HOST_BYTES_BASE:
                fails.append(f"{part}: host-concat budget {budget} is below the measured "
                             f"{tt.CONCAT_HOST_BYTES_BASE} base — the budget may only widen")
            if dram <= 12 * 2 ** 30 and budget != 1_610_612_736:
                fails.append(f"{part}: a {dram / 2**30:.3f} GiB part must keep the measured "
                             f"1.5 GiB host-concat budget byte for byte, got {budget}")
            if dram >= 16 * 2 ** 30 and budget <= 1_610_612_736:
                fails.append(f"{part}: {dram / 2**30:.3f} GiB of DRAM and the host-concat budget "
                             f"is still the 12 GiB-Wormhole figure {budget} — the issue-#11 "
                             f"class, a calibration point applied outside its measured range")
            for aa, h, host_on_wh in CONCAT_REFINER_SHAPES:
                row["checks"] += 1
                on_host = h * h * 384 * 2 > budget
                want = host_on_wh and dram <= 12 * 2 ** 30
                if on_host != want:
                    fails.append(f"{part}: the OpenDDE refiner pair tensor at {aa} aa (H={h}, "
                                 f"{h * h * 384 * 2 / 2**30:.4f} GiB) assembles on the "
                                 f"{'host' if on_host else 'device'} and should be on the "
                                 f"{'host' if want else 'device'}")

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
        table_grids = {g for _n, g, _l, _d, _p in L1_BUDGET_PARTS}
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
    for name, grid, _l1, _dram, _prov in L1_BUDGET_PARTS:
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


def batch_position_failures(rows: list, n: int) -> list:
    """Why a batch-position run fails, or [] if it passes. Pure, so it is testable.

    ``rows[:n]`` are byte-identical targets and must agree on every field in
    BATCH_POSITION_FIELDS. ``rows[n:]`` is the differing control, which must NOT match
    them: without that check a run where every fold collapsed to one constant would also
    report agreement, which is the right answer for the wrong reason.
    """
    same = rows[:n]
    if len(same) < n:
        return [f"only {len(same)}/{n} identical targets folded"]
    fails = []
    for f in BATCH_POSITION_FIELDS:
        vals = [r.get(f) for r in same]
        if len(set(map(repr, vals))) != 1:
            fails.append(f"{f} depends on batch position: {vals}")
    control = rows[n:]
    if not control:
        fails.append("no differing control target in the batch, so agreement is unproven")
    else:
        for f in BATCH_POSITION_FIELDS:
            if control[0].get(f) == same[0].get(f):
                fails.append(f"the control target matched the identical ones on {f} "
                             f"({control[0].get(f)}), so this leg is not discriminating")
    return fails


def run_batch_position(keep: bool) -> dict:
    """Fold identical targets at different batch positions and require identical results.

    Drives scripts/boltz2_affinity_batch_position_repro.py, which folds every target
    through the same ``_WorkerState`` path a spawned worker uses and writes one JSON
    record per target. Its own exit code already encodes the identical-target verdict;
    this leg re-derives it from the JSON so a partial run cannot read as a pass, and
    adds the differing-control check.
    """
    row = {"model": "batch-position", "seconds": None, "rows": [], "gate": False,
           "error": None}
    if not BATCH_POSITION_SCRIPT.exists():
        row["error"] = f"missing {BATCH_POSITION_SCRIPT}"
        return row
    out = REPO_ROOT / "batch_position_gate"
    if out.exists():
        shutil.rmtree(out)
    js = out / "probe.json"
    out.mkdir(parents=True)
    cmd = [sys.executable, str(BATCH_POSITION_SCRIPT),
           "--aa", str(BATCH_POSITION_AA), "--n", str(BATCH_POSITION_N),
           "--extra-aa", str(BATCH_POSITION_EXTRA_AA),
           "--out", str(out), "--json-out", str(js)]
    print(f"\n{'='*70}\n[batch-position] {BATCH_POSITION_N} identical "
          f"{BATCH_POSITION_AA} aa targets + one {BATCH_POSITION_EXTRA_AA} aa control, "
          f"one process\n{'='*70}", flush=True)
    log = REPO_ROOT / "batch_position_gate.log"
    # Each target is a full structure + affinity fold, so this leg needs N+1 of them.
    timeout = FOLD_TIMEOUT_S * (BATCH_POSITION_N + 1)
    t0 = time.monotonic()
    with open(log, "wb") as fh:
        rc, timed_out = _run_fold(cmd, timeout, cwd=REPO_ROOT,
                                  stdout=fh, stderr=subprocess.STDOUT)
    row["seconds"] = time.monotonic() - t0
    if timed_out:
        row["error"] = f"repro timed out after {timeout}s (see {log})"
        return row
    if not js.exists():
        row["error"] = f"repro wrote no JSON (rc={rc}); see {log}"
        return row
    row["rows"] = json.loads(js.read_text())
    fails = batch_position_failures(row["rows"], BATCH_POSITION_N)
    if rc and not fails:
        fails.append(f"repro exited {rc}; see {log}")
    row["gate"] = not fails
    row["error"] = "; ".join(fails) or None
    if not keep:
        shutil.rmtree(out, ignore_errors=True)
    return row


def main() -> int:
    # Scorers, folds and predict CLIs we spawn arm their parent-death guard off this,
    # so none of them can outlive this driver still holding a card. Inherited through
    # every spawn path in this file (see tt_bio/device_lease.py:arm_orphan_guard).
    os.environ["TT_BIO_PARENT_PID"] = str(os.getpid())
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    choices=list(MODELS) + list(DEFAULT_ARMS) + ["size-ladder"]
                    + ESMC_DEFAULT + ESMC_OPT_IN,
                    action="append",
                    help="Gate only this model (repeatable). Default: the five fold "
                         "models + boltzgen + rfd3 + opendde-abag + rf3-1024aa + "
                         "capacity + l1-budget + size-ladder + ESMC 300m/600m embed "
                         "parity. esmc-6b is opt-in (slow ~13 GB load).")
    ap.add_argument("--keep", action="store_true", help="Keep run output dirs for inspection.")
    ap.add_argument("--size-ladder-record-lever", default=None, metavar="FLAG[,FLAG...]",
                    help="Add new census levers to the existing size-ladder baseline "
                         "instead of re-recording it: one fold per (model, rung), and the "
                         "splice is refused unless every other lever still matches. For a "
                         "counter-only lever, which changes no behaviour and does not "
                         "justify discarding a good timing baseline.")
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
    ap.add_argument("--load-ceiling", type=float, default=gate_guard.DEFAULT_LOAD_CEILING,
                    help="Refuse to start when the 1-min loadavg is above this multiple of "
                         f"nproc (default {gate_guard.DEFAULT_LOAD_CEILING}; 0 disables). Every "
                         "leg here folds in a subprocess, so a gate started on an already-"
                         "loaded box both measures noise and helps overcommit the host.")
    args = ap.parse_args()
    global FAST, DIFFUSION_TRACE
    FAST = args.fast
    DIFFUSION_TRACE = args.diffusion_trace

    # This gate is single-card by construction: every leg folds one yaml, which selects one
    # device, and the boltzgen leg passes --devices <first granted card>. So there is nothing
    # here to skip for a narrow grant; what it does need is the load guard, and the grant
    # printed so a run's card is in its own log rather than inferred from the launch line.
    overloaded = gate_guard.load_ceiling_problem(args.load_ceiling)
    if overloaded:
        print(f"PREFLIGHT - refusing to run the gate. {overloaded}")
        return 1
    print(f"[release-gate] granted {gate_guard.grant_label(gate_guard.card_grant())}", flush=True)

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

    models = args.model or list(MODELS) + list(DEFAULT_ARMS) + ["size-ladder"] + ESMC_DEFAULT
    fold_models = [m for m in models if m in MODELS]
    want_boltzgen = "boltzgen" in models
    want_opendde_abag = "opendde-abag" in models
    want_rfd3 = "rfd3" in models
    want_pxdesign = "pxdesign" in models
    want_capacity = "capacity" in models
    want_l1_budget = "l1-budget" in models
    want_batch_position = "batch-position" in models
    want_nesso1 = "nesso1" in models
    want_rf3_1024aa = "rf3-1024aa" in models
    want_size_ladder = "size-ladder" in models
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
              f"{_steps_label(fold_models)} steps / {DIFFUSION_SAMPLES} samples, "
              f"seed {SEED}\n{'#'*78}")
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

    if want_rf3_1024aa:
        rr = run_rf3_1024aa(args.keep)
        print(f"\n{'#'*78}\nRELEASE GATE — {RF3_1024AA_FIXTURE} (rf3, 997 aa), "
              f"{_sampling_steps('rf3')} steps / 1 sample, seed {RF3_1024AA_SEED}"
              f"\n{'#'*78}")
        print(f"{'model':<15}{'vs xtal (A)':>13}{'n CA':>6}{'ref':>8}{'X':>8}"
              f"{'floor':>10}{'wall':>9}  result")
        xt = f"{rr['xtal_a']:.3f}" if rr["xtal_a"] is not None else "  -  "
        nca = str(rr["n_ca"]) if rr["n_ca"] is not None else "-"
        rx = f"{rr['ref_xtal_a']:.3f}" if rr["ref_xtal_a"] is not None else "  -  "
        xa = f"{rr['x_a']:.3f}" if rr["x_a"] is not None else "  -  "
        wall = f"{rr['seconds']:.0f}s" if rr["seconds"] is not None else "-"
        verdict = "PASS" if rr["gate"] else f"FAIL ({rr['error']})" if rr["error"] else "FAIL"
        all_pass &= rr["gate"]
        print(f"{rr['model']:<15}{xt:>13}{nca:>6}{rx:>8}{xa:>8}"
              f"{f'<={RF3_1024AA_MAX_XTAL_A}':>10}{wall:>9}  {verdict}")
        print(f"ref = the reference's own distance to the crystal, X = device vs reference: "
              f"both evidence, not gated (measured {RF3_1024AA_XTAL_MEASURED} A)")
        print(f"{'#'*78}")
        print("GATE PASS — rf3 at 997 aa cleared the crystal floor" if rr["gate"]
              else "GATE FAIL — rf3 at 997 aa missed the crystal floor (see above)")

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

    if want_rfd3:
        if not RFD3_SPEC.exists():
            sys.exit(f"missing rfd3 gate spec {RFD3_SPEC}")
        rr = run_rfd3(args.keep)
        print(f"\n{'#'*78}\nRELEASE GATE — {RFD3_SPEC.name} (rfd3), "
              f"{RFD3_NUM_DESIGNS} designs, {RFD3_TIMESTEPS} steps, seed {RFD3_SEED}"
              f"\n{'#'*78}")
        print(f"{'model':<15}{'designs':>8}{'clean':>7}{'in band':>9}{'breaks':>7}"
              f"{'clash':>12}{'AA':>5}{'UNK':>5}{'det':>5}{'wall':>9}  result")
        ib = f"{rr['in_band']:.4f}" if rr["in_band"] is not None else "  -  "
        cf = (f"{rr['clashes']}({rr['clash_frac']:.4f})"
              if rr["clashes"] is not None else "  -  ")
        br = str(rr["breaks"]) if rr["breaks"] is not None else "-"
        aa = str(rr["distinct_aa"]) if rr["distinct_aa"] is not None else "-"
        unk = str(rr["unk"]) if rr["unk"] is not None else "-"
        det = ("ok" if rr["determinism"] else "NO") if rr["determinism"] is not None else "-"
        wall = f"{rr['seconds']:.0f}s" if rr["seconds"] is not None else "-"
        verdict = "PASS" if rr["gate"] else f"FAIL ({rr['error']})" if rr["error"] else "FAIL"
        all_pass &= rr["gate"]
        cr = f"{rr['clean_rate']:.2f}" if rr["clean_rate"] is not None else "  -  "
        print(f"{rr['model']:<15}{rr['n_designs']:>8}{cr:>7}{ib:>9}{br:>7}{cf:>12}{aa:>5}"
              f"{unk:>5}{det:>5}{wall:>9}  {verdict}")
        print(f"floor{'':<10}{RFD3_NUM_DESIGNS:>8}{RFD3_MIN_CLEAN_RATE:>7.2f}"
              f"{RFD3_MIN_INBAND:>9.4f}{RFD3_MAX_BREAKS:>7}{RFD3_MAX_CLASHES:>12}"
              f"{RFD3_MIN_DISTINCT_AA:>5}{RFD3_MAX_UNK:>5}{'ok':>5}")
        print(f"{'#'*78}")
        print("GATE PASS — rfd3 designs cleared parse, designed-region geometry, "
              "sequence and determinism" if rr["gate"]
              else "GATE FAIL — rfd3 missed parse, geometry, sequence or determinism (see above)")

    if want_pxdesign:
        pr = run_pxdesign(args.keep)
        print(f"\n{'#'*78}\nRELEASE GATE — {PXDESIGN_SPEC.name} (pxdesign), "
              f"{PXDESIGN_NUM_DESIGNS} design, {PXDESIGN_N_STEP} steps\n{'#'*78}")
        print(f"{'model':<15}{'fit RMSD':>12}{'binder res':>12}{'floor':>18}{'wall':>9}  result")
        floor = f"<={PXDESIGN_MAX_FIT_RMSD}A"
        fit = f"{pr['fit_rmsd']:.3f}" if pr["fit_rmsd"] is not None else "  -  "
        res = str(pr["binder_residues"]) if pr["binder_residues"] is not None else "  -  "
        wall = f"{pr['seconds']:.0f}s" if pr["seconds"] is not None else "-"
        verdict = "PASS" if pr["gate"] else f"FAIL ({pr['error']})" if pr["error"] else "FAIL"
        all_pass &= pr["gate"]
        print(f"{pr['model']:<15}{fit:>12}{res:>12}{floor:>18}{wall:>9}  {verdict}")
        if pr["sha16"]:
            same = "  MATCH" if pr["sha16"] == PXDESIGN_STRUCTURE_SHA16 else "  DIFFERS"
            print(f"coordinate digest {pr['sha16']}{same}  (evidence, not gated — a digest is "
                  f"card- and arch-specific)")
        print(f"{'#'*78}")
        print("GATE PASS — pxdesign designs cleared parse + the conditioning floor" if pr["gate"]
              else "GATE FAIL — pxdesign missed parse or the conditioning floor (see above)")

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

    if want_nesso1:
        nr = run_nesso1(args.keep)
        print(f"\n{'#'*78}\nRELEASE GATE — {NESSO1_FIXTURE.name} (nesso1), "
              f"{NESSO1_TRUNK} trunk / {NESSO1_AFFINITY} affinity, "
              f"{NESSO1_REPEATS} device repeats\n{'#'*78}")
        print(f"{'model':<15}{'worst vs ref':>14}{'dev spread':>12}{'floor':>18}"
              f"{'wall':>9}  result")
        fl = nr.get("floors") or {}
        floor = (f"<={fl.get('max_x_over_r', '?')}xR/<={fl.get('max_device_spread', '?'):g}"
                 if fl else "-")
        xr = f"{nr['x_over_r']:.3f}xR" if nr["x_over_r"] is not None else "  -  "
        sp = f"{nr['spread']:.3g}" if nr["spread"] is not None else "  -  "
        wall = f"{nr['seconds']:.0f}s" if nr["seconds"] is not None else "-"
        verdict = "PASS" if nr["gate"] else f"FAIL ({nr['error']})" if nr["error"] else "FAIL"
        all_pass &= nr["gate"]
        print(f"{nr['model']:<15}{xr:>14}{sp:>12}{floor:>18}{wall:>9}  {verdict}")
        print(f"{'#'*78}")
        print("GATE PASS — nesso1 scalars cleared the reference floor and the device is "
              "deterministic" if nr["gate"]
              else "GATE FAIL — nesso1 missed the reference floor or drifted run to run (see above)")

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

    if want_size_ladder and args.size_ladder_record_lever:
        sl = run_size_ladder_add_lever(args.size_ladder_record_lever, args.keep,
                                       Path(args.size_ladder_baseline),
                                       args.size_ladder_models.split(",")
                                       if args.size_ladder_models else None)
        rows.append(sl)
        all_pass &= sl["gate"]
        # Printed, and folded into all_pass. Without this the mode appended its row and
        # returned silently: a model whose splice was REFUSED looked identical to one that
        # was never asked for, which is the failure mode this whole arm exists to not have.
        print(f"\n{'#'*78}\nRELEASE GATE — size-ladder add-lever "
              f"{args.size_ladder_record_lever} (card {sl.get('card', '?')})\n{'#'*78}")
        for l in sl["legs"]:
            added = l.get("added") or {}
            what = ("added at " + ",".join(sorted(added, key=int)) + " aa" if added
                    else f"REFUSED ({l['error']})" if l.get("error") else "REFUSED")
            print(f"{l['model']:<15}{'PASS' if l['gate'] else 'FAIL':<6}{what}")
            for f in (l.get("findings") or []):
                print(f"    FAIL {f}")
        if not sl["legs"] and sl.get("error"):
            print(f"    FAIL {sl['error']}")
        print(f"{'#'*78}")
        print("LEVERS ADDED — fill every TODO exemption reason in "
              f"{args.size_ladder_baseline}, then run the arm without "
              f"--size-ladder-record-lever" if sl["gate"] else
              "ADD-LEVER REFUSED — something other than the named levers moved, so a full "
              "--size-ladder-record is the honest fix (see above)")
    elif want_size_ladder:
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
        # Exponent columns are the intervals the arm actually gates, i.e. consecutive
        # pairs of SIZE_LADDER_EXP_RUNGS -- NOT of every rung. Pairing consecutive rungs
        # printed a k512->640 / k640->768 that is never computed and left k512->768, the
        # one interval with a real tolerance, without a column at all.
        exp_rungs = [n for n in rungs if n in SIZE_LADDER_EXP_RUNGS]
        intervals = list(zip(exp_rungs, exp_rungs[1:]))
        hdr = f"{'model':<15}" + "".join(f"{str(n) + 'aa':>9}" for n in rungs)
        hdr += "".join(f"{f'k{a}->{b}':>11}" for a, b in intervals)
        hdr += f"{'wall':>9}  result"
        print(hdr)
        for l in sl["legs"]:
            rt = l.get("runtime_s") or {}
            ex = l.get("exponents") or {}
            cells = "".join(f"{(f'{rt[str(n)]:.1f}s' if rt.get(str(n)) is not None else '-'):>9}"
                            for n in rungs)
            for a, b in intervals:
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

    if want_batch_position:
        bp = run_batch_position(args.keep)
        all_pass &= bp["gate"]
        print(f"\n{'#'*78}\nRELEASE GATE — batch position "
              f"({BATCH_POSITION_N} identical {BATCH_POSITION_AA} aa CDK2 + SMILES-ligand "
              f"targets, then one {BATCH_POSITION_EXTRA_AA} aa control, one process)"
              f"\n{'#'*78}")
        print(f"{'pos':<5}{'target':<12}{'coords':>18}{'affinity':>11}"
              f"{'p(bind)':>10}{'wall':>9}")
        for r in bp["rows"]:
            wall = f"{r['wall_s']:.0f}s" if r.get("wall_s") is not None else "-"
            print(f"{r.get('pos', '-'):<5}{r.get('target', '-'):<12}"
                  f"{str(r.get('coords', '-')):>18}"
                  f"{str(r.get('affinity_pred_value', '-')):>11}"
                  f"{str(r.get('affinity_probability_binary', '-')):>10}{wall:>9}")
        if bp["error"]:
            print(f"    FAIL {bp['error']}")
        print(f"{'#'*78}")
        print("GATE PASS — identical targets are identical whatever their batch position, "
              "and the control target still differs" if bp["gate"] else
              "GATE FAIL — a result depends on a target's position in the batch (see above)")

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
