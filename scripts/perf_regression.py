#!/usr/bin/env python3
"""Per-model performance regression gate — the perf leg of RELEASING.md.

Measures WARM steady-state throughput for every shipped model on a fixed small
input, compares it to a committed per-model baseline (``docs/perf_baselines.json``),
and FAILS if a model regressed beyond a configurable noise margin. Designed to run
before every release so a perf regression can't ship silently.

What it measures, per model:

  * fold models (boltz2, esmfold2, esmfold2-fast, protenix-v2, opendde) —
    structures/s on ``examples/trpcage.yaml`` (20 aa), single-sequence, 1 recycling
    cycle / 10 sampling steps / 1 sample. The model is loaded ONCE in-process
    (``tt_bio.worker._WorkerState``), one warmup fold absorbs the first-kernel
    compile, then N timed folds give a warm median. Model load + first-compile are
    EXCLUDED — this is a dispatch/throughput number, not a cold-start or production
    fold. It catches kernel/dispatch regressions, not accuracy (that is
    ``scripts/release_gate.py``'s job).
  * esmc-300m / esmc-600m embed — seq/s on a fixed batch of 8 ubiquitin-length
    sequences (batch_size 8). Same warmup-then-time protocol.
  * esmc-6b embed — seq/s on 8 ubiquitin-length sequences. The 6B backbone
    (sharded TransformerEngine, ~13 GB resident) runs one-sequence-at-a-time
    (embed_sequences ignores batch_size for the 6B -- no room to widen the batch),
    so batch_size is nominal (1) and the timed work is 8 sequential forwards.
    Same warmup-then-time protocol; a different runtime shape than 300m/600m,
    gated separately so a 6B dispatch/throughput regression can't ship silently.
  * boltzgen — designs/s on ``examples/binder.yaml`` (protein-anything, 4
    designs). A single end-to-end ``tt-bio gen run`` subprocess (design +
    inverse-fold + refold + analysis + filter); the first design's first-kernel
    compile is included in the timed region, so this is a conservative
    cold-inflated warm-throughput proxy. Reuses the SAME fixture/protocol the
    designability accuracy leg gates.
  * boltz2-affinity — affinities/s on ``examples/affinity_fkg.yaml``
    (FKBP12+SB3, L107, single-seq, ``--affinity_mw_correction``). A single
    end-to-end ``tt-bio predict`` subprocess in Boltz-2's binding-affinity mode
    (README "Binding Affinity Prediction"): fold the complex, then re-run the
    affinity model's own 64-block trunk + AtomDiffusion + affinity heads from
    ``boltz2_aff.ckpt``. The first call's first-kernel compile is included in the
    timed region, so this is a conservative cold-inflated warm-throughput proxy
    (same character as the boltzgen leg — the affinity path has no warm
    steady-state loop to repeat). Reuses the SAME fixture the affinity accuracy
    leg (docs/implementation-parity.md) folds. Shipped-default fp32 host gates stay
    ON (no env overrides) so the timed call matches the shipped config.
  * saprot-650m embed — seq/s on a fixed batch of 8 ubiquitin-length sequences
    (batch_size 8). Device-resident ESM-2 over the fused AA+Foldseek-3Di vocab,
    loaded via ``tt_bio.saprot`` directly (the worker's embed path is
    ESMC-specific). Same warmup-then-time protocol as esmc-300m/600m; sequence-only
    mode (3Di="#"), no foldseek on the perf path.

Baselines live in ``docs/perf_baselines.json`` and are EXPLICIT and PER-CARD-TYPE
with a PER-MACHINE layer under that. The file nests one block per card type
(``p150a``, ``p300c``, ...) under a ``cards`` key; each card block carries a
card-level ``models`` map (the fallback) AND an optional ``machines`` map whose
keys are physical-machine ids (``socket.gethostname()``, the repo's existing
convention in ``tt_bio/runtime.py``) each pointing at a machine-specific
``models`` map. The gate resolves a model's baseline as
``cards.<card_type>.machines.<machine_id>.models.<model>`` if a machine-specific
entry exists for the detected machine, else falls back to
``cards.<card_type>.models.<model>`` — so the scheme is backward compatible and
does NOT require every card type to carry a full per-machine block (only the
models that actually differ per machine need one). The gate detects the card it
is running on at runtime via tt-smi / kernel sysfs, mirroring
``tt_bio/main.py::_detect_p300_devices``. A P300c baseline must never be judged
against a P150a run — the P150 is a smaller chip and would read as a false 20-34%
regression that is just the card, not the code. A baseline seeded on one physical
p150a (e.g. ``pc``) must not be judged against a run on a different physical p150a
(e.g. ``qb1``, ~30-36% slower on the same models) — the machine-id layer guards
that within-card-type machine variance. If the detected card type has no
recorded baseline at all, the gate FAILS loudly (every model NO BASELINE) rather
than silently skipping or matching the wrong card's numbers. An intentional perf
change (landed optimization, deliberate accuracy/perf tradeoff) updates the
baseline via ``--update-baseline --note "<why>"`` (writes to the detected
machine's machine-specific block) — never silently. A regression the author
didn't intend fails the gate. Cover new models / new card types as they ship by
adding a spec here + a baseline entry (seeded on that card type / machine).

Regression threshold (default 15%) — evidence, not a guess. Run-to-run spread
was measured on qb2 (p300c) by running the gate 3x per model as separate
invocations (fresh process, fresh first-kernel compile each time):

  * embed  (esmc-300m, warm median of 5): 33.506 / 33.531 / 33.506 seq/s  → 0.08%
  * fold   (boltz2,    warm median of 5): 1.524 / 1.520 / 1.519 struct/s  → 0.34%
  * single-shot (boltz2-affinity, 1 timed): 72.1 s / 74.1 s wall          → ~2.7%

The warm-median legs (fold/embed) are extremely stable (<0.5%) because WARMUP
absorbs compile and the median of REPEAT smooths dispatch jitter. The single-shot
legs (kind="gen"/"affinity") have NO warm loop to median over — one cold-inflated
wall-clock — so they are the noisy floor (~3% here; the gen pipeline, longer and
unmeasured, is expected to be similar or a little higher). 15% sits comfortably
above that ~3% worst-case floor with margin for the gen leg and for thermal/clock
drift over the weeks between releases, while still catching any material
kernel/dispatch regression (which shows up as tens of percent, not single digits).
Do NOT tighten below ~10% without adding a warm loop to the single-shot legs
first, or they will false-alarm on their own run-to-run noise.

Usage::

    # run the whole gate on the card (one device context per model subprocess)
    TT_VISIBLE_DEVICES=0 PYTHONPATH=<worktree> python3 scripts/perf_regression.py

    # one model / a subset
    python3 scripts/perf_regression.py --model boltz2 --model esmfold2

    # seed / refresh baselines from the current warm runs (explicit, needs a note)
    python3 scripts/perf_regression.py --update-baseline --note "seed from 0.2.5 main"

    # custom regression threshold (percent; default 15)
    python3 scripts/perf_regression.py --threshold 10

Exit 0 iff every requested model is within threshold of its baseline (or, with
``--update-baseline``, every model measured successfully). 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_FILE = REPO_ROOT / "docs" / "perf_baselines.json"

# Fixed, tiny, deterministic inputs — small enough that the timed region is a
# few seconds per fold, so the whole gate runs in minutes. trpcage (20 aa) is the
# canonical fast fold target; the embed batch is 8x ubiquitin (76 aa).
TRPCAGE = REPO_ROOT / "examples" / "trpcage.yaml"
# BoltzGen's canonical binder-design fixture (de-novo binder vs chain A of 7ROA,
# protein-anything protocol) — the SAME target README documents for `tt-bio gen run`
# and the designability accuracy leg (scripts/release_gate.py) gates. Reused here
# for the perf leg so the two legs share one fixture, not two.
BINDER = REPO_ROOT / "examples" / "binder.yaml"
UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")  # 76 aa — tests/test_esmc.py / scripts/esmc_embed_parity.py golden

# Boltz-2 binding-affinity fixture (FKBP12 + SB3, L107, ) — the SAME
# target docs/implementation-parity.md's affinity accuracy leg folds. Reused here for
# the perf/UX legs so all three legs share one fixture, not three. The affinity
# path is a heavier runtime shape than a structure fold: it folds the complex
# (conf model) AND re-runs the affinity model's own 64-block trunk + atom
# diffusion + affinity heads from a separate boltz2_aff.ckpt. The shipped default
# (BOLTZ2_AFFINITY_TRUNK_FP32_HOST=1, BOLTZ2_AFFINITY_FP32_HOST=1,
# BOLTZ2_AFFINITY_DIFFUSION_FP32_HOST=0) runs the affinity pairformer + heads in
# fp32 on host and the 64-block affinity trunk in fp32 on host (~140 s of the
# ~170 s per-target wall-clock); the gate times the shipped default — no env
# overrides — so the number reflects what a customer experiences.
AFFINITY = REPO_ROOT / "examples" / "affinity_fkg.yaml"

# ── card-type detection ────────────────────────────────────────────────────
# The gate is card-type aware: a P300c baseline must NOT be compared against a
# P150a run — the P150 is a smaller chip, so the same code reads as a 20-34%
# "regression" that is just the card, not the code (found 2026-07-14 by the
# hardware-limit recheck). The per-card baseline key is the canonical board_type
# tt-smi reports (p150a / p300c / ...). Detection mirrors
# tt_bio/main.py::_detect_p300_devices (kernel sysfs, no device open) so it is
# cheap and runs in the parent before any model loads; tt-smi names boards sysfs
# can't and is the canonical source when available.
_P300_SUBSYSTEMS = {"0x0044", "0x0045", "0x0046"}  # Blackhole P300 (lone-chip custom topology)

# Per-model measurement spec. ``kind`` is "fold" or "embed". ``unit`` + ``direction``
# define the gated metric (throughput, higher is better). Every fold model uses the
# same light protocol (1 recycle / 10 steps / 1 sample) so numbers are comparable
# across releases and the gate stays fast.
SPECS: dict[str, dict] = {
    "boltz2":         dict(kind="fold", unit="structures/s", direction="higher"),
    "esmfold2":       dict(kind="fold", unit="structures/s", direction="higher"),
    "esmfold2-fast":  dict(kind="fold", unit="structures/s", direction="higher"),
    "protenix-v2":    dict(kind="fold", unit="structures/s", direction="higher"),
    "opendde":        dict(kind="fold", unit="structures/s", direction="higher"),
    "esmc-300m":      dict(kind="embed", unit="seq/s", direction="higher",
                           batch_size=8, n_seqs=8),
    "esmc-600m":      dict(kind="embed", unit="seq/s", direction="higher",
                           batch_size=8, n_seqs=8),
    # ESMC-6B is the sharded-TransformerEngine LM backbone (~13 GB resident
    # weights). embed_sequences runs it one-sequence-at-a-time -- the 6B forward
    # already buckets and its weight footprint leaves no room to widen the
    # batch (see tt_bio.esmc.embed_sequences) -- so batch_size is nominal (1)
    # and the timed work is n_seqs sequential ubiquitin forwards. Same
    # embed-kind protocol shape as 600m (warmup-then-time, seq/s, higher=better)
    # so a dispatch/throughput regression on the 6B load path can't ship
    # silently -- it has no entry otherwise and is a different runtime shape
    # than 300m/600m.
    "esmc-6b":        dict(kind="embed", unit="seq/s", direction="higher",
                           batch_size=1, n_seqs=8),
    # SaProt-650M is the flagship structure-aware protein-LM checkpoint (ESM-2
    # over the fused AA+Foldseek-3Di vocab, 446 tokens; tt_bio/saprot.py). It is
    # device-resident like esmc-300m/600m, so it mirrors that embed shape exactly
    # (batch_size=8, n_seqs=8 ubiquitin, seq/s, warmup-then-time). Sequence-only
    # mode (3Di="#") -- no foldseek dependency on the perf path. The 35M/1.3B
    # sizes are skipped here, matching how only two ESMC sizes are gated. The
    # worker's embed path is ESMC-specific, so the measurement loads via
    # tt_bio.saprot directly (see _measure_saprot_embed) -- same warm seq/s
    # protocol, just a different loader than esmc.
    "saprot-650m":    dict(kind="embed", unit="seq/s", direction="higher",
                           batch_size=8, n_seqs=8),
    "boltzgen":       dict(kind="gen", unit="designs/s", direction="higher",
                           num_designs=4, protocol="protein-anything"),
    # Boltz-2 binding-affinity prediction mode (README "Binding Affinity
    # Prediction" — affinity_prediction=True, the affinity model's own 64-block
    # trunk + AtomDiffusion re-run + affinity heads, distinct from structure
    # prediction). A real customer-facing CLI mode () that
    # had ZERO perf-gate coverage. kind="affinity" is a single end-to-end CLI
    # subprocess like the gen leg (the affinity path has no warm steady-state
    # predict_one loop — it folds once then predicts affinity once per target),
    # so warmup=0/repeat=1 and the gated metric is affinities/s = 1 / wall-clock.
    # The first call's first-kernel compile is included in the timed region, so
    # this is a conservative cold-inflated warm-throughput proxy — same character
    # as the boltzgen leg. Shipped-default fp32 host gates stay ON (no env
    # overrides) so the timed call matches the shipped config.
    "boltz2-affinity": dict(kind="affinity", unit="affinities/s", direction="higher"),
}
DEFAULT_MODELS = list(SPECS)

# Light fold protocol — fast, exercises the full trunk + diffusion + heads path.
RECYCLING_STEPS = 1
SAMPLING_STEPS = 10
DIFFUSION_SAMPLES = 1
WARMUP = 2          # warmup folds absorb first-kernel compile (excluded from timing)
REPEAT = 5          # timed folds; report the median
DEFAULT_THRESHOLD = 15.0   # % regression allowed before the gate fails; see docstring
                           # "regression threshold" note for the measured evidence.

# Wall-clock ceilings so a wedged device / hung dependency can never stall a
# release (the same standing rule the gate redesign established: every long step
# gets a timeout + an honest fallback — a timeout is reported as a measurement
# FAILURE, which is itself a gate failure, never a silent skip). Generous vs the
# real cost (an in-process fold gate is a model load + WARMUP+REPEAT tiny folds,
# minutes; the gen leg is a full 4-design production pipeline) so a timeout means
# genuinely stuck, not merely slow. Env-overridable for a slow host.
MEASURE_TIMEOUT_S = int(os.environ.get("PERF_MEASURE_TIMEOUT", "1800"))   # fold/embed/affinity child
GEN_TIMEOUT_S = int(os.environ.get("PERF_GEN_TIMEOUT", "3600"))           # full design pipeline


# ── baseline file ──────────────────────────────────────────────────────────

def load_baselines() -> dict:
    if not BASELINE_FILE.exists():
        return {"cards": {}}
    return json.loads(BASELINE_FILE.read_text())


def save_baselines(data: dict) -> None:
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _sysfs_subsystem_device(device_id: str) -> str | None:
    """Read the PCI subsystem_device for one tenstorrent device from kernel sysfs.
    Same source as tt_bio/main.py::_detect_p300_devices — no device open, no tt-smi."""
    for entry in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            did = entry.name.rsplit("!", 1)[1]
        except Exception:
            continue
        if did != device_id:
            continue
        try:
            return (entry / "device" / "subsystem_device").read_text().strip().lower()
        except Exception:
            return None
    return None


def _resolve_tt_smi() -> str | None:
    """Absolute path to the ``tt-smi`` CLI, or None if it can't be found.

    The gate must not depend on the caller's PATH: under non-interactive ssh
    ``~/.local/bin`` (where tt-smi lives on the release hosts) is typically NOT
    on PATH, so a bare ``subprocess.run(["tt-smi", ...])`` silently fails, the
    gate falls back to sysfs, misdetects the board, and compares against the
    wrong baseline. Resolve tt-smi from an explicit known-good path list
    (PATH first, then ``~/.local/bin`` and the system bins) and call it by
    absolute path so detection is identical whether or not ``~/.local/bin`` is
    on PATH.
    """
    found = shutil.which("tt-smi")
    if found:
        return found
    for c in (
        # Next to the running interpreter first: on the release hosts tt-smi is
        # pip-installed into the same venv as tt-bio (e.g. <env>/bin), which is
        # NOT on PATH under non-interactive ssh and is NOT ~/.local/bin. Missing
        # this was a live misdetection risk — the gate fell through to the sysfs
        # board map, which only knows a fixed subsystem set, so any board not in
        # that set read as 'unknown' (NO BASELINE) even though tt-smi was installed
        # and would have named it. NOTE: do NOT .resolve() sys.executable — a venv
        # python is a symlink to the system interpreter, so resolving it would point
        # at /usr/bin and miss the venv's own tt-smi. Use the unresolved bin dir.
        Path(sys.executable).parent / "tt-smi",
        Path.home() / ".local" / "bin" / "tt-smi",
        Path("/usr/local/bin/tt-smi"),
        Path("/usr/bin/tt-smi"),
    ):
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def detect_card_type() -> str:
    """Canonical board-type key for the card this gate will run on ('p150a',
    'p300c', ...). This is the per-card baseline key in docs/perf_baselines.json.
    No device is opened; safe to call in the parent before any model loads.

    tt-smi is resolved by absolute path (see ``_resolve_tt_smi``) so detection
    does not depend on the caller's PATH. If tt-smi can't be found the gate
    falls back to sysfs and reports a recognizable ``unknown:<sub>`` key so it
    fails loudly against a missing baseline instead of silently matching the
    wrong one; a stderr warning points the operator at PATH.
    """
    visible = (os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    # Primary: tt-smi -s reports the canonical board_type — matches the baseline
    # key exactly and names boards the sysfs subsystem map can't.
    tt_smi = _resolve_tt_smi()
    if tt_smi is not None:
        try:
            out = subprocess.run([tt_smi, "-s"], capture_output=True, text=True,
                                 timeout=20, check=False)
            info = json.loads(out.stdout).get("device_info", [])
            if info:
                idx = min(int(visible), len(info) - 1) if visible.isdigit() else 0
                bt = info[idx].get("board_info", {}).get("board_type")
                if bt:
                    return str(bt).lower()
        except Exception:
            pass
    else:
        print(f"{sys.argv[0]}: WARNING: tt-smi not found on PATH or in "
              f"~/.local/bin; card detection falling back to sysfs and may "
              f"report 'unknown' (NO BASELINE). Add tt-smi to PATH (e.g. "
              f"export PATH=$HOME/.local/bin:$PATH) and re-run.",
              file=sys.stderr)
    # Fallback: sysfs subsystem_device -> known Blackhole board types. An
    # unrecognized subsystem returns a recognizable 'unknown:<sub>' key so the
    # gate fails loudly against a missing baseline instead of silently matching
    # the wrong one.
    sub = _sysfs_subsystem_device(visible)
    if sub in _P300_SUBSYSTEMS:
        return "p300c"
    if sub:
        return f"unknown:{sub}"
    return "unknown"


def detect_machine_id() -> str:
    """Stable per-machine key for the machine-id baseline layer
    (``cards.<card_type>.machines.<machine_id>.models``). Reuses the repo's
    existing hostname convention (``tt_bio/runtime.py::build_local_workers``,
    ``tt_bio/main.py``) so a machine is identified the same way here and in
    worker-slot naming: ``socket.gethostname()``.
    """
    return socket.gethostname()


def card_baselines(data: dict, card_type: str, machine_id: str | None = None) -> dict | None:
    """The resolved per-model baseline map for ``card_type`` on ``machine_id``,
    or None if this card type has no recorded baseline at all (the gate must
    fail loudly on that).

    Two-level lookup with backward-compatible fallback: a model's baseline is
    taken from ``cards.<card_type>.machines.<machine_id>.models.<model>`` if a
    machine-specific entry exists for the detected machine, otherwise from
    ``cards.<card_type>.models.<model>`` (today's shape). This guards
    within-card-type machine variance — e.g. qb1's p150a cards read ~30-36%
    slower than pc's p150a on the SAME models, so a baseline seeded on pc reads
    as a false regression on qb1 (and vice versa). A machine-specific entry
    overrides the card-level fallback per model, so a card type does NOT need a
    full per-machine block — only the models that actually differ per machine
    need one. If no machine-specific entry exists for the detected machine the
    gate falls back to the card-level block unchanged (today's behavior)."""
    cards = data.get("cards")
    if not cards and data.get("models"):
        # Legacy single-card file (pre per-card split) — treat it as one card so
        # an un-updated checkout still gates instead of crashing.
        return data["models"]
    entry = cards.get(card_type) if cards else None
    if not entry:
        return None
    card_models = entry.get("models", {})
    if not machine_id:
        return card_models
    machines = entry.get("machines")
    if not machines:
        return card_models
    m_entry = machines.get(machine_id)
    if not m_entry:
        return card_models
    machine_models = m_entry.get("models", {})
    if not machine_models:
        return card_models
    # Machine-specific overrides card-level per model; models only in the
    # card-level block fall through unchanged (backward-compatible fallback).
    return {**card_models, **machine_models}


def _version() -> str:
    import re
    txt = (REPO_ROOT / "pyproject.toml").read_text()
    m = re.search(r'^version\s*=\s*"([^"]+)"', txt, re.M)
    return m.group(1) if m else "unknown"


# ── in-process measurement (runs in a child subprocess, one device context) ─

def _boltz_conf_kwargs() -> tuple[dict, dict]:
    """Build Boltz-2's load-time conf_kwargs with the light perf protocol.

    Boltz-2 bakes recycling/sampling/diffusion_samples into the model at load
    (predict_args) rather than reading them from cfg at fold time, so they must
    be set here. Mirrors tt_bio/main.py's predict path; steering off (no
    potentials), kernels on, TT on, trace off."""
    _diffusion = {"step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                  "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0,
                  "sigma_data": 16.0, "P_mean": -1.2, "P_std": 1.5,
                  "coordinate_augmentation": True, "alignment_reverse_diff": True,
                  "synchronize_sigmas": True}
    _pairformer = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
    _msa = {"subsample_msa": False, "num_subsampled_msa": 1024, "use_paired_feature": True}
    predict_args = {"recycling_steps": RECYCLING_STEPS, "sampling_steps": SAMPLING_STEPS,
                    "diffusion_samples": DIFFUSION_SAMPLES, "max_parallel_samples": 1}
    steering = {"fk_steering": False, "physical_guidance_update": False,
                "contact_guidance_update": False, "num_particles": 3, "fk_lambda": 4.0,
                "fk_resampling_interval": 3, "num_gd_steps": 20}
    conf = dict(
        predict_args=predict_args, diffusion_process_args=_diffusion,
        pairformer_args=_pairformer, msa_args=_msa, steering_args=steering,
        use_kernels=True, use_tenstorrent=True, trace=False, diffusion_trace=False,
    )
    aff = dict(predict_args={**predict_args, "recycling_steps": 5, "max_parallel_samples": 1},
               diffusion_process_args=_diffusion, pairformer_args=_pairformer, msa_args=_msa,
               steering_args=steering, affinity_mw_correction=False, use_tenstorrent=True,
               trace=False, diffusion_trace=False)
    return conf, aff


def _build_cfg(model: str, spec: dict, struct_dir: Path, msa_dir: Path) -> dict:
    cfg = dict(
        model=model,
        fast=False,
        output_format="cif",
        recycling_steps=RECYCLING_STEPS,
        sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES,
        seed=0,
        trace=False,
        msa_dir=str(msa_dir),
        struct_dir=str(struct_dir),
        use_msa_server=False,
        msa_db_path=None,
        use_envdb=False,
        msa_endpoint=None,
        single_sequence=True,        # fold single-seq: no MSA, no network, deterministic
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        msa_server_username=None,
        msa_server_password=None,
        api_key_value=None,
        max_msa_seqs=8192,
        write_pae=False,
        write_pde=False,
        write_embeddings=False,
        method=None,
        # esmc embed fields (ignored by fold models)
        pool="mean",
        batch_size=spec.get("batch_size", 8),
        return_logits=False,
    )
    if model == "boltz2":
        cfg["conf_kwargs"], cfg["aff_kwargs"] = _boltz_conf_kwargs()
    return cfg


def _write_embed_fasta(path: Path, n_seqs: int) -> None:
    lines = []
    for i in range(n_seqs):
        lines.append(f">seq{i}|protein")
        lines.append(UBIQUITIN)
    path.write_text("\n".join(lines) + "\n")


def _log_tail(log_path: Path) -> str:
    """Last non-empty line of a subprocess log, for a one-line failure summary."""
    if not Path(log_path).exists():
        return ""
    lines = Path(log_path).read_text().strip().splitlines()
    return lines[-1] if lines else ""


def _run_cli(cmd: list[str], env: dict, log_path: Path, timeout: int, label: str) -> float:
    """Run a tt-bio CLI subprocess to completion, timed, with a hard wall-clock
    timeout, and return the wall-clock seconds.

    Spawns in its own session (``start_new_session``) so a timeout kills the
    WHOLE process tree via ``killpg`` — a ``tt-bio predict``/``gen`` fans out
    device workers, and a bare ``subprocess`` timeout would SIGKILL only the
    launcher and orphan those workers still holding the card (wedging every later
    leg). Raises ``RuntimeError`` (with the log tail) on a non-zero exit or a
    timeout so the caller records a measurement FAILURE — the standing gate rule:
    no leg hangs forever on a flaky dependency; a timeout is a gate failure, never
    a silent skip."""
    import signal
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
            wall = time.perf_counter() - t0
            raise RuntimeError(
                f"{label} exceeded {timeout}s timeout (killed after {wall:.0f}s) — "
                f"treat as a measurement failure / possible device wedge")
    wall = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(f"{label} exited {rc} after {wall:.0f}s: {_log_tail(log_path)}")
    return wall


def _measure_gen(model: str, spec: dict, out_path: Path) -> dict:
    """Time one ``tt-bio gen run`` design job end-to-end and write a JSON result.

    BoltzGen is a *design* pipeline, not a fold loop: it has no warm steady-state
    ``predict_one`` to repeat. So this leg spawns the shipping ``tt-bio gen run``
    CLI as a subprocess (the pipeline owns its own device lifecycle — no device is
    opened in this measure process) and times the full design + inverse-fold +
    refold + analysis + filter pipeline on the canonical binder fixture. The
    gated metric is ``designs/s = num_designs / wall-clock``.

    This is a single end-to-end invocation, not a warm loop: the first design
    absorbs first-kernel compile and is included in the timed region, so
    ``designs/s`` is a conservative (cold-inflated) warm-throughput proxy. The cold
    fraction is stable across releases, so a dispatch/throughput regression still
    shows up as a higher wall-clock. Reuses the SAME fixture/protocol the
    designability accuracy leg runs (``examples/binder.yaml``,
    ``protein-anything``) — no new fixture invented.
    """
    spec_path = BINDER
    if not spec_path.exists():
        raise FileNotFoundError(f"missing gen fixture {spec_path}")
    n = spec["num_designs"]
    protocol = spec["protocol"]
    work = Path(tempfile.mkdtemp(prefix=f"perf-{model}-"))
    out_dir = work / "gen"
    log_path = work / "gen.log"
    cmd = [
        sys.executable, "-m", "tt_bio.main", "gen", "run", str(spec_path),
        "--output", str(out_dir),
        "--num_designs", str(n),
        "--protocol", protocol,
        "--devices", "1",
        "--budget", str(n),
        "--debug",  # headless: no Rich live view, no-op reporter
    ]
    env = dict(os.environ)
    pp = str(REPO_ROOT)
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    env.setdefault("LOGURU_LEVEL", "WARNING")
    wall = _run_cli(cmd, env, log_path, GEN_TIMEOUT_S, f"gen run [{model}]")
    throughput = n / wall
    latency_ms = wall / n * 1000.0
    card = detect_card_type()
    result = dict(
        model=model,
        kind=spec["kind"],
        unit=spec["unit"],
        direction=spec["direction"],
        hardware="blackhole",
        card_type=card,
        throughput=round(throughput, 6),
        latency_ms=round(latency_ms, 2),
        median_s=round(wall, 4),
        times_s=[round(wall, 4)],
        load_s=0.0,
        warmup=0,
        repeat=1,
        # protein-anything production defaults (design 500 / refold 200 steps,
        # recycling 3) — informational; the perf leg does not override them.
        sampling_steps=500,
        diffusion_samples=1,
        recycling_steps=3,
        num_designs=n,
        protocol=protocol,
        input=f"{spec_path.name} ({protocol}, {n} designs)",
        tt_bio_version=_version(),
        date=date.today().isoformat(),
    )
    out_path.write_text(json.dumps(result))
    print(f"[{model}] {result['throughput']} {spec['unit']}  "
          f"({latency_ms:.0f} ms/design, wall {wall:.0f}s)", file=sys.stderr)
    shutil.rmtree(work, ignore_errors=True)
    return result


def _measure_affinity(model: str, spec: dict, out_path: Path) -> dict:
    """Time one ``tt-bio predict`` affinity-mode call end-to-end and write a JSON
    result.

    Boltz-2's binding-affinity mode (README "Binding Affinity Prediction") is a
    real customer-facing CLI path that had no perf-gate coverage. It is a heavier
    runtime shape than a structure fold: it folds the complex (conf model) AND
    re-runs the affinity model's own 64-block trunk + AtomDiffusion + affinity
    heads from a separate boltz2_aff.ckpt. There is no warm steady-state
    ``predict_one`` loop to repeat (one target = fold-once + predict-affinity-once),
    so this leg mirrors the gen leg: a single end-to-end ``tt-bio predict``
    subprocess (the CLI owns its device lifecycle; no device is opened in this
    measure process) timed wall-to-wall. The gated metric is
    ``affinities/s = 1 / wall-clock``.

    The first call's first-kernel compile is included in the timed region, so
    ``affinities/s`` is a conservative (cold-inflated) warm-throughput proxy; the
    cold fraction is stable across releases, so a dispatch/throughput regression
    still shows up as a higher wall-clock. Uses the SAME FKBP12+SB3 fixture the
    affinity accuracy leg (docs/implementation-parity.md) folds, with a light sampling
    protocol (1 structure recycle / 10 structure steps / 1 structure sample +
    10 affinity steps / 1 affinity sample) so the gate stays in minutes while
    exercising the full affinity path. The shipped-default fp32 host gates
    (BOLTZ2_AFFINITY_TRUNK_FP32_HOST=1, BOLTZ2_AFFINITY_FP32_HOST=1,
    BOLTZ2_AFFINITY_DIFFUSION_FP32_HOST=0) are left at their defaults (no env
    overrides) so the timed call matches the shipped config.
    """
    spec_path = AFFINITY
    if not spec_path.exists():
        raise FileNotFoundError(f"missing affinity fixture {spec_path}")
    work = Path(tempfile.mkdtemp(prefix=f"perf-{model}-"))
    out_dir = work / "out"
    log_path = work / "affinity.log"
    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(spec_path),
        "--model", "boltz2",
        "--single_sequence",
        "--override",
        "--affinity_mw_correction",
        "--debug",  # NullDisplay: headless, no Rich TTY
        "--recycling_steps", str(RECYCLING_STEPS),
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--sampling_steps_affinity", str(SAMPLING_STEPS),
        "--diffusion_samples_affinity", str(DIFFUSION_SAMPLES),
        "--out_dir", str(out_dir),
    ]
    env = dict(os.environ)
    pp = str(REPO_ROOT)
    env["PYTHONPATH"] = pp + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    env.setdefault("LOGURU_LEVEL", "WARNING")
    wall = _run_cli(cmd, env, log_path, MEASURE_TIMEOUT_S, f"affinity predict [{model}]")
    throughput = 1.0 / wall
    latency_ms = wall * 1000.0
    card = detect_card_type()
    result = dict(
        model=model,
        kind=spec["kind"],
        unit=spec["unit"],
        direction=spec["direction"],
        hardware="blackhole",
        card_type=card,
        throughput=round(throughput, 6),
        latency_ms=round(latency_ms, 2),
        median_s=round(wall, 4),
        times_s=[round(wall, 4)],
        load_s=0.0,
        warmup=0,
        repeat=1,
        sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES,
        recycling_steps=RECYCLING_STEPS,
        sampling_steps_affinity=SAMPLING_STEPS,
        diffusion_samples_affinity=DIFFUSION_SAMPLES,
        input=f"{spec_path.name} (FKBP12+SB3, L107, single-seq, affinity mode)",
        tt_bio_version=_version(),
        date=date.today().isoformat(),
    )
    out_path.write_text(json.dumps(result))
    print(f"[{model}] {result['throughput']} {spec['unit']}  "
          f"({latency_ms:.0f} ms/call, wall {wall:.0f}s)", file=sys.stderr)
    shutil.rmtree(work, ignore_errors=True)
    return result


def measure(model: str, out_path: Path) -> dict:
    """Load one model, warmup, time REPEAT folds, write a JSON result to out_path.

    Runs in its own subprocess (see _run_measure) so each model gets a fresh
    device context — model weights are released cleanly and we avoid the
    cross-model device-reopen path that the worker loop deliberately never takes.
    """
    spec = SPECS[model]
    if spec["kind"] == "gen":
        return _measure_gen(model, spec, out_path)
    if spec["kind"] == "affinity":
        return _measure_affinity(model, spec, out_path)
    if model.startswith("saprot"):
        return _measure_saprot_embed(model, spec, out_path)
    import torch  # noqa: F401  — imported by worker anyway; sets grad off below
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name, cleanup
    from tt_bio.worker import _WorkerState, _ensure_local_artifacts
    from tt_bio import esmfold2 as _E
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor

    _noop = lambda *a, **k: None
    _E.set_progress(_noop)

    # A lone P300 Blackhole chip is a custom topology: ttnn refuses to open it
    # without a 1x1 mesh-graph descriptor. The predict/embed CLIs set this per
    # worker — mirror them here so a direct get_device() works on a P300 box.
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    get_device()  # open the chip once for this process
    hw = arch_name()
    card = detect_card_type()

    work = Path(tempfile.mkdtemp(prefix=f"perf-{model}-"))
    struct_dir = work / "out"
    msa_dir = work / "msa"
    struct_dir.mkdir(parents=True, exist_ok=True)
    msa_dir.mkdir(parents=True, exist_ok=True)

    if spec["kind"] == "fold":
        if not TRPCAGE.exists():
            raise FileNotFoundError(f"missing fold input {TRPCAGE}")
        input_path = TRPCAGE
    else:
        input_path = work / "embed.fasta"
        _write_embed_fasta(input_path, spec["n_seqs"])

    cfg = _build_cfg(model, spec, struct_dir, msa_dir)
    _ensure_local_artifacts(cfg)

    state = _WorkerState("tenstorrent")
    t_load = time.perf_counter()
    state.load_model(cfg)
    load_s = time.perf_counter() - t_load
    state.bind_run("perf", cfg)
    state.pfn = _noop
    if cfg["model"] == "boltz2":
        state.model.progress_fn = _noop

    job_cfg = dict(cfg)

    def one_fold():
        job_cfg["struct_dir"] = str(struct_dir)
        # wipe between calls so saving stays inside the timed region but never
        # short-circuits on a stale output (predict_one overwrites anyway)
        for p in struct_dir.glob("*"):
            p.unlink()
        t0 = time.perf_counter()
        metrics, _best, _feats = state.predict_one(input_path, job_cfg)
        return time.perf_counter() - t0

    # warmup — absorbs first-kernel compile; never timed
    for _ in range(WARMUP):
        one_fold()

    times = []
    for _ in range(REPEAT):
        times.append(one_fold())

    times.sort()
    median = times[len(times) // 2]
    if spec["kind"] == "fold":
        throughput = 1.0 / median               # structures/s (one structure per fold)
        latency_ms = median * 1000.0
    else:
        n = spec["n_seqs"]
        throughput = n / median                 # seq/s (one batched forward per call)
        latency_ms = median * 1000.0

    result = dict(
        model=model,
        kind=spec["kind"],
        unit=spec["unit"],
        direction=spec["direction"],
        hardware=hw,
        card_type=card,
        throughput=round(throughput, 6),
        latency_ms=round(latency_ms, 2),
        median_s=round(median, 4),
        times_s=[round(t, 4) for t in times],
        load_s=round(load_s, 1),
        warmup=WARMUP,
        repeat=REPEAT,
        sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES,
        recycling_steps=RECYCLING_STEPS,
        input="trpcage (20 aa, single-seq)" if spec["kind"] == "fold"
              else f"{spec['n_seqs']}x ubiquitin (76 aa), batch {spec['batch_size']}",
        tt_bio_version=_version(),
        date=date.today().isoformat(),
    )
    out_path.write_text(json.dumps(result))
    print(f"[{model}] {result['throughput']} {spec['unit']}  "
          f"({latency_ms:.0f} ms/call, load {load_s:.0f}s)", file=sys.stderr)

    state.reset()
    cleanup()
    return result


def _measure_saprot_embed(model: str, spec: dict, out_path: Path) -> dict:
    """In-process warm-throughput measurement for SaProt embed (device-resident
    ESM-2 over the fused AA+Foldseek-3Di vocab).

    Mirrors the esmc-300m/600m embed leg (batch_size=8, n_seqs=8 ubiquitin,
    seq/s, warmup-then-time) but loads via tt_bio.saprot directly: the
    worker's embed path (_predict_embed_one) is ESMC-specific (it calls
    tt_bio.esmc.embed_sequences / load_sequences), and SaProt has its own
    loader / tokenizer / embed_sequences (see scripts/saprot_parity.py for
    the same direct-load pattern). Sequence-only mode (3Di="#"), so no foldseek
    dependency on the perf path. Same warm seq/s protocol as esmc embed, just a
    different loader — a dispatch/throughput regression on the SaProt load path
    can't ship silently.
    """
    import torch
    torch.set_grad_enabled(False)
    from tt_bio.tenstorrent import get_device, arch_name, cleanup
    from tt_bio.main import _detect_p300_devices, _find_ttnn_mesh_graph_descriptor
    from tt_bio import saprot

    # P300 lone-chip workaround — must be set before the first get_device() call
    # (Saprot.from_pretrained opens the device in TorchWrapper.__init__).
    if _detect_p300_devices() and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        mgd = _find_ttnn_mesh_graph_descriptor("p150_mesh_graph_descriptor.textproto")
        if mgd:
            os.environ["TT_MESH_GRAPH_DESC_PATH"] = mgd

    get_device()
    hw = arch_name()
    card = detect_card_type()

    work = Path(tempfile.mkdtemp(prefix=f"perf-{model}-"))
    fasta = work / "embed.fasta"
    _write_embed_fasta(fasta, spec["n_seqs"])
    seqs = saprot.load_sequences_with_structure(str(fasta), None)

    t_load = time.perf_counter()
    m = saprot.load_saprot(model)
    load_s = time.perf_counter() - t_load

    def one_call():
        t0 = time.perf_counter()
        saprot.embed_sequences(m, seqs, pool="mean", batch_size=spec["batch_size"])
        return time.perf_counter() - t0

    for _ in range(WARMUP):
        one_call()
    times = [one_call() for _ in range(REPEAT)]
    times.sort()
    median = times[len(times) // 2]
    n = spec["n_seqs"]
    throughput = n / median                 # seq/s (one batched forward per call)
    latency_ms = median * 1000.0

    result = dict(
        model=model,
        kind=spec["kind"],
        unit=spec["unit"],
        direction=spec["direction"],
        hardware=hw,
        card_type=card,
        throughput=round(throughput, 6),
        latency_ms=round(latency_ms, 2),
        median_s=round(median, 4),
        times_s=[round(t, 4) for t in times],
        load_s=round(load_s, 1),
        warmup=WARMUP,
        repeat=REPEAT,
        sampling_steps=SAMPLING_STEPS,
        diffusion_samples=DIFFUSION_SAMPLES,
        recycling_steps=RECYCLING_STEPS,
        input=f"{spec['n_seqs']}x ubiquitin (76 aa), batch {spec['batch_size']}",
        tt_bio_version=_version(),
        date=date.today().isoformat(),
    )
    out_path.write_text(json.dumps(result))
    print(f"[{model}] {result['throughput']} {spec['unit']}  "
          f"({latency_ms:.0f} ms/call, load {load_s:.0f}s)", file=sys.stderr)

    cleanup()
    shutil.rmtree(work, ignore_errors=True)
    return result


# ── parent: spawn one subprocess per model, compare, report ────────────────

def _run_measure(model: str) -> dict | None:
    """Run the per-model measurement in a fresh subprocess (one device context).

    Each model gets its own process so model weights are released cleanly and we
    avoid the cross-model device-reopen path the worker loop deliberately never
    takes (see tt_bio/worker.py run_worker_loop)."""
    td = tempfile.mkdtemp(prefix="perf-out-")
    out = Path(td) / "result.json"
    cmd = [
        sys.executable, str(Path(__file__).resolve()), "--measure", model,
        "--out", str(out),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + env["PYTHONPATH"]
                                          if env.get("PYTHONPATH") else "")
    env.setdefault("TT_VISIBLE_DEVICES", "0")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    env.setdefault("LOGURU_LEVEL", "WARNING")
    # Parent-side backstop timeout: the inner CLI legs (gen/affinity) already
    # bound their own grandchild, so this is the ceiling for the in-process
    # fold/embed child (which has no inner subprocess) plus a margin over the
    # gen ceiling. start_new_session + killpg so a wedge reaps the whole tree,
    # not just the launcher (which would orphan device-holding workers).
    import signal
    timeout = (GEN_TIMEOUT_S if SPECS[model]["kind"] == "gen" else MEASURE_TIMEOUT_S) + 300
    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    try:
        rc = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            proc.kill()
        proc.wait()
        print(f"[{model}] measurement TIMED OUT after {timeout}s — killed the "
              f"process tree; treating as a gate failure (possible device wedge)",
              file=sys.stderr)
        return None
    if rc != 0 or not out.exists():
        print(f"[{model}] measurement FAILED (exit {rc})", file=sys.stderr)
        return None
    try:
        return json.loads(out.read_text())
    except Exception as e:
        print(f"[{model}] failed to parse result: {e}", file=sys.stderr)
        return None


def _delta_str(baseline: float, current: float, direction: str) -> tuple[float, str]:
    if direction == "higher":
        pct = (current - baseline) / baseline * 100.0
    else:
        pct = (baseline - current) / baseline * 100.0
    sign = "+" if pct >= 0 else ""
    return pct, f"{sign}{pct:.1f}%"


def _passes(baseline: float, current: float, direction: str, threshold: float) -> bool:
    pct, _ = _delta_str(baseline, current, direction)
    return pct >= -threshold


def _print_table(rows: list[dict], baselines: dict, card_type: str, machine_id: str,
                    threshold: float) -> bool:
    """Print the per-model comparison table. Returns True iff every row passes.

    Compares each model against the baseline resolved for ``card_type`` on
    ``machine_id`` — a P300c baseline must never be judged against a P150a run
    (the P150 is a smaller chip and would read as a false 20-34% regression),
    and a baseline seeded on one physical p150a (e.g. pc) must not be judged
    against a run on a different physical p150a (e.g. qb1, ~30-36% slower) —
    the machine-id layer under card type guards that within-type variance. If
    no baseline exists at all for the detected card type, the gate FAILS loudly
    (every model NO BASELINE) rather than silently skipping or matching the
    wrong card's numbers."""
    all_pass = True
    bm = card_baselines(baselines, card_type, machine_id)
    have_card = bm is not None
    # The warm-protocol suffix is per-row (fold/embed legs use WARMUP+REPEAT; the
    # gen leg is a single end-to-end pipeline run, warmup=0/repeat=1). Describe
    # the first row's protocol so the title never mislabels a gen-only run as
    # "2 warmup + 5 timed".
    r0 = rows[0] if rows else {}
    w = r0.get("warmup", WARMUP)
    rep = r0.get("repeat", REPEAT)
    warm_desc = (f"warm ({w} warmup + {rep} timed)" if r0.get("kind") not in ("gen", "affinity")
                 else f"single end-to-end run ({rep} timed)")
    title = (f"PERF REGRESSION GATE — card {card_type} @ {machine_id} — "
             f"{', '.join(r['model'] for r in rows)}  "
             f"| threshold ±{threshold:.0f}%  | {warm_desc}")
    print(f"\n{'#' * 78}\n{title}\n{'#' * 78}")
    hdr = (f"{'model':<16}{'metric':<16}{'baseline':>11}{'current':>11}"
           f"{'delta':>10}{'verdict':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        unit = r["unit"]
        if not have_card:
            cur = f"{r['throughput']:.4g}" if not r.get("failed") else "FAILED"
            print(f"{r['model']:<16}{unit:<16}{'(none)':>11}{cur:>11}{'n/a':>10}{'NO BASELINE':>10}")
            all_pass = False
            continue
        b = bm.get(r["model"])
        if b is None:
            cur = f"{r['throughput']:.4g}" if not r.get("failed") else "FAILED"
            print(f"{r['model']:<16}{unit:<16}{'(none)':>11}{cur:>11}{'n/a':>10}{'NO BASELINE':>10}")
            all_pass = False
            continue
        if r.get("failed"):
            print(f"{r['model']:<16}{unit:<16}{float(b['value']):>11.4g}{'FAILED':>11}{'n/a':>10}{'FAIL':>10}")
            all_pass = False
            continue
        base = float(b["value"])
        pct, delta = _delta_str(base, r["throughput"], r["direction"])
        ok = _passes(base, r["throughput"], r["direction"], threshold)
        all_pass &= ok
        verdict = "PASS" if ok else "FAIL"
        print(f"{r['model']:<16}{unit:<16}{base:>11.4g}{r['throughput']:>11.4g}"
              f"{delta:>10}{verdict:>10}")
    print("-" * len(hdr))
    print(f"  card: {card_type}  |  machine: {machine_id}  |  hardware: {rows[0].get('hardware', '?')}  "
          f"|  tt-bio {rows[0].get('tt_bio_version', '?')}  |  input: {rows[0].get('input', '?')}")
    if not have_card:
        msg = (f"GATE FAIL — no baseline recorded for card type '{card_type}' in "
               f"{BASELINE_FILE.relative_to(REPO_ROOT)}. Seed it on a {card_type} "
               f"card with: python3 scripts/perf_regression.py --update-baseline "
               f"--note \"seed {card_type} baseline\"")
    else:
        msg = ("GATE PASS — no model regressed beyond ±{:.0f}%".format(threshold) if all_pass
               else "GATE FAIL — a model regressed beyond ±{:.0f}% (see above)".format(threshold))
    print(f"{'#' * 78}\n{msg}")
    return all_pass


def cmd_gate(args) -> int:
    models = args.model or DEFAULT_MODELS
    rows = []
    for m in models:
        if m not in SPECS:
            sys.exit(f"unknown model {m!r}; choose from {', '.join(SPECS)}")
        r = _run_measure(m)
        if r is None:
            # a measurement failure is itself a gate failure
            rows.append(dict(model=m, unit=SPECS[m]["unit"], direction=SPECS[m]["direction"],
                             throughput=0.0, hardware="?", tt_bio_version="?",
                             input="?", failed=True))
            continue
        r["failed"] = False
        rows.append(r)

    if args.update_baseline:
        return _update_baselines(rows, args)

    baselines = load_baselines()
    card_type = detect_card_type()
    machine_id = detect_machine_id()
    ok = _print_table(rows, baselines, card_type, machine_id, args.threshold)
    return 0 if ok else 1


def _update_baselines(rows: list[dict], args) -> int:
    if not args.note:
        sys.exit("--update-baseline requires --note \"<why this perf change is intended>\"")
    data = load_baselines()
    cards = data.setdefault("cards", {})
    card_type = detect_card_type()
    machine_id = detect_machine_id()
    entry = cards.setdefault(card_type, {})
    # Write to the machine-specific block (cards.<card_type>.machines.<machine_id>.models)
    # so a baseline is tagged with the physical machine that produced it — the gate
    # resolves a model's baseline from the machine block first and falls back to the
    # card-level ``models`` block if no machine-specific entry exists (see
    # ``card_baselines``). This guards within-card-type machine variance (e.g. qb1 vs
    # pc p150a, ~30-36% delta) without requiring every card type to carry a full
    # per-machine block.
    machines = entry.setdefault("machines", {})
    m_entry = machines.setdefault(machine_id, {})
    models = m_entry.setdefault("models", {})
    any_ok = False
    for r in rows:
        if r.get("failed"):
            print(f"[{r['model']}] FAILED — not updating its baseline", file=sys.stderr)
            continue
        any_ok = True
        models[r["model"]] = dict(
            unit=r["unit"], direction=r["direction"], value=r["throughput"],
            latency_ms=r["latency_ms"], input=r["input"],
            sampling_steps=r["sampling_steps"], diffusion_samples=r["diffusion_samples"],
            recycling_steps=r["recycling_steps"], warmup=r["warmup"], repeat=r["repeat"],
            hardware=r["hardware"], card_type=r.get("card_type", card_type),
            machine_id=machine_id,
            tt_bio_version=r["tt_bio_version"], date=r["date"], note=args.note,
        )
        m_entry["date"] = r["date"]
        m_entry["tt_bio_version"] = r["tt_bio_version"]
        m_entry["note"] = args.note
    # Drop a legacy top-level "models" so the file is unambiguously per-card.
    data.pop("models", None)
    data["hardware"] = data.get("hardware", "blackhole")
    data["threshold_pct"] = args.threshold
    save_baselines(data)
    machine_names = {ct: sorted(ct_entry.get("machines", {})) for ct, ct_entry in cards.items()}
    machine_names = {ct: ms for ct, ms in machine_names.items() if ms}
    print(f"\nWrote {BASELINE_FILE.relative_to(REPO_ROOT)}  "
          f"(card {card_type} @ {machine_id}: {len(models)} models; "
          f"{len(cards)} card type(s) recorded: {', '.join(sorted(cards))}; "
          f"machines: " +
          ", ".join(f"{ct}=[{', '.join(ms)}]" for ct, ms in machine_names.items()) + ")")
    print("Review the diff, then commit it with the change that justifies the new numbers.")
    return 0 if any_ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append", choices=list(SPECS),
                    help="Gate only this model (repeatable). Default: all shipped models.")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Regression %% allowed before FAIL (default {DEFAULT_THRESHOLD:g}).")
    ap.add_argument("--update-baseline", action="store_true",
                    help="Refresh docs/perf_baselines.json from these warm runs instead of "
                         "gating. Requires --note. Use for an INTENTIONAL perf change only.")
    ap.add_argument("--note", default=None,
                    help="Required with --update-baseline: why this perf change is intended.")
    # internal: the per-model in-process measurement subprocess
    ap.add_argument("--measure", metavar="MODEL", help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.measure:
        if args.measure not in SPECS:
            sys.exit(f"unknown model {args.measure!r}")
        if args.out is None:
            sys.exit("--out is required with --measure")
        try:
            measure(args.measure, args.out)
            return 0
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[{args.measure}] measurement error: {e}", file=sys.stderr)
            return 1

    return cmd_gate(args)


if __name__ == "__main__":
    sys.exit(main())
