#!/usr/bin/env python3
"""UX-regression release gate — the user-experience leg of RELEASING.md.

Complements ``scripts/release_gate.py`` (accuracy) and the perf gate. This leg
does NOT measure accuracy or speed — it asserts the user-facing *plumbing* every
release ships with still works, headlessly and fast, on a tiny input:

  1. LIVE PROGRESS VIEW — for every fold model the streamed progress events
     advance through every real phase (load → trunk recycling iterations →
     diffusion steps → done) with no phase skipped. This is exactly the guard
     against the "0 → diffusion" / "loading → diffusion" jump class of bugs
     fixed by the predict-progress-fix work. It drives a headless JSONL event
     capture (``TT_BIO_PROGRESS_CAPTURE=<path>``) teed off the *same* event
     stream the live Rich view reads in ``_stream_run``, so it observes real
     predict behaviour — not a scraped TTY, not a synthetic replay.
  2. OUTPUT FILES PARSE — the emitted CIF (fold models) / npz (esmc embed)
     load under a strict standard parser (``Bio.PDB.MMCIFParser`` /
     ``numpy.load``), catching the malformed-output class (e.g. the historical
     missing ``_atom_site.occupancy`` fixed in 17aeab9e).
  3. CLI behaves — ``tt-bio predict --help`` / ``tt-bio embed --help`` /
     ``tt-bio saprot --help`` / the unified ``tt-bio design --help`` (with
     ``--model rfd3|boltzgen``) / the deprecated ``tt-bio gen run --help``
     alias (exit 0 AND a deprecation warning on stderr) all exit 0 and list
     the core flags, and each surface's results/manifest file has the shape
     the downstream reader expects.

Coverage is DISCOVERED, not listed. The gated set comes from the ``*_MODELS``
tuples in ``tt_bio.main`` — the single source of truth every ``--model`` choice
list is built from — and ``_assert_full_model_coverage`` refuses to start if a
shipped model has neither a leg nor a written reason in ``LEGS_EXEMPT``. The
hardcoded lists this replaced are how nesso1, openbind and pxdesign all reached
0.7.0 with zero UX coverage while the gate stayed green. Same idiom as
``scripts/perf_regression.py:_assert_full_model_coverage`` and
``scripts/packaging_smoke.py:_expected_data_files``.

The legs, by shape:

  * fold — every ``predict --model`` choice, on examples/trpcage.yaml
    (opendde-abag on the Ab-Ag fixture examples/1ahw_abag.yaml), legs 1–3.
  * embed — every ``embed --model`` and ``saprot --model`` choice, legs 2–3.
    Embed has no fold phases; its progress is the load → embed → done stdout
    lines.
  * design — boltzgen via ``tt-bio design --model boltzgen``, whose progress is
    the pipeline's own stdout stage stream under ``--debug --log``; rfd3 and
    pxdesign via ``tt-bio design --model <m>``, whose progress is the plain
    Designing → per-design → Done stdout lines.
  * affinity — nesso1 via ``tt-bio affinity`` (a scalar screen: no coordinates,
    so the legs are its stdout phases plus the two files it writes), and
    Boltz-2's affinity mode via ``tt-bio predict --affinity_mw_correction``,
    whose event stream is the same shape as a structure fold, so that leg reuses
    _check_progress plus an affinity_pred_value results check.

Fast + deterministic: folds ``examples/trpcage.yaml`` (20 residues; opendde-abag
uses the larger 1ahw_abag Ab-Ag complex) with ``recycling_steps=2``,
``sampling_steps=4``, ``diffusion_samples=1``, ``--single_sequence`` for the
MSA-dependent models. This checks UX plumbing, not accuracy — it does not need
full folds. Exit 0 iff every requested leg PASSES; 1 otherwise. Runs on the
device serially (one card context per predict).

    # gate every surface on card 0 (run with the project venv, like release_gate)
    TT_VISIBLE_DEVICES=0 /path/to/env/bin/python scripts/ux_regression.py
    # one model
    /path/to/env/bin/python scripts/ux_regression.py --model boltz2
    /path/to/env/bin/python scripts/ux_regression.py --model esmc-600m
    # CLI-behaviour leg only (no card needed — usable in GitHub CI)
    /path/to/env/bin/python scripts/ux_regression.py --cli-only
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tt_bio import main as tt_bio_main, weights  # noqa: E402  (after the path insert)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gate_guard  # noqa: E402  (interpreter guard, shared with the two release gates)

REPO_ROOT = Path(__file__).resolve().parent.parent
# trpcage (20 residues) is the canonical tiny fold target — small enough that
# even the ESMC-6B ESMFold2 load dominates wall-clock, so the gate stays fast.
DATA = REPO_ROOT / "examples" / "trpcage.yaml"
NAME = DATA.stem  # "trpcage" -> predict writes <model>_results_trpcage/

# Minimal step counts: enough to prove the trunk and diffusion phases each tick
# (≥1 event with total>0), not enough to matter for accuracy. UX plumbing only.
RECYCLING_STEPS = 2
SAMPLING_STEPS = 4
DIFFUSION_SAMPLES = 1
SEED = 0
# Per-model wall-clock budget. Load dominates; trpcage is tiny, but ESMFold2
# (ESMC-6B ~12.8 GB) and Protenix-v2 (~1.9 GB ckpt) take a few minutes to load.
# opendde-abag folds the larger Ab-Ag fixture (1ahw_abag, ~440 residues), so it
# gets a looser budget than the trpcage fold models.
PER_MODEL_TIMEOUT_S = 900
ABAG_MODEL_TIMEOUT_S = 1800

# Every `predict --model` choice gets a fold leg, taken from the CLI's own choice
# list. A new fold model is covered the day it ships; there is no list here to
# forget, which is exactly how openbind shipped with no UX leg at all.
FOLD_MODELS = list(tt_bio_main.PREDICT_MODELS)
# MSA-dependent models get --single_sequence so the gate is offline + deterministic
# (no ColabFold server round-trip). Read from the same tuple the CLI itself uses to
# decide whether a model wants an MSA, so a new MSA model cannot be handed the wrong
# flag here. esmfold2 / esmfold2-fast are single-sequence by design and absent from it.
MSA_DEPENDENT = set(tt_bio_main.MSA_DEFAULT_MODELS) & set(FOLD_MODELS)
# opendde-abag is the antibody-antigen checkpoint, so it is gated on the canonical
# Ab-Ag fixture 1ahw_abag.yaml (the same SAbDab/PDB 1ahw target the benchmark uses
# elsewhere) instead of trpcage. Every other fold model uses trpcage.
ABAG_DATA = REPO_ROOT / "examples" / "1ahw_abag.yaml"
# Every embed choice, across both subcommands: `tt-bio embed` for esmc-*, `tt-bio
# saprot` for saprot-* (SaProt has its own CLI entry, not the esmc embed command).
# Both write the same npz + manifest.json shape, so the parse/manifest checks are
# shared and _embed_subcommand picks the verb. Discovered, so a new checkpoint size
# is gated on arrival rather than exempted on the grounds that it "shares the path"
# — the reasoning that let opendde-abag ship with no perf coverage.
EMBED_MODELS = list(tt_bio_main.EMBED_MODELS) + list(tt_bio_main.SAPROT_MODELS)

# BoltzGen (binder design) — exercised via `tt-bio design --model boltzgen` on
# the canonical binder fixture (same target the designability accuracy leg + the
# perf leg use).
# A tiny 1-design job is enough to gate the UX plumbing (progress phases, output
# parses, CLI shape); it is not an accuracy or perf measurement.
GEN_MODEL = "boltzgen"
GEN_SPEC = REPO_ROOT / "examples" / "binder.yaml"
GEN_PROTOCOL = "protein-anything"
GEN_NUM_DESIGNS = 1
GEN_TIMEOUT_S = 1200  # design + refold + analysis for 1 design; load dominates

# Boltz-2 binding-affinity prediction mode (README "Binding Affinity Prediction")
# — exercised via `tt-bio predict examples/affinity_fkg.yaml --model boltz2
# --affinity_mw_correction`. A real customer-facing CLI mode that had ZERO UX-gate
# coverage. The affinity path's progress event stream is the SAME shape as a
# structure fold (loading → msa → prep → trunk → diffusion → confidence → saving
# → done): the affinity model re-runs its OWN 64-block trunk + AtomDiffusion
# after the structure fold, but that re-run is silent (no progress_fn is wired to
# the affinity model), so the live view advances through the structure phases
# then completes. Verified by capturing a real affinity run's event stream. So
# this leg reuses the fold leg's _check_progress (the must-never-recur bug class
# — a progress bar jumping past a phase instead of advancing phase-by-phase) on
# the affinity path specifically, plus an affinity-specific results check: the
# user-facing affinity_pred_value scalar must be present in results.json.
AFFINITY_MODEL = "boltz2-affinity"
AFFINITY_SPEC = REPO_ROOT / "examples" / "affinity_fkg.yaml"  # FKBP12+SB3, L107, msa: empty
AFFINITY_TIMEOUT_S = 900  # affinity trunk fp32 (5 recycles, 64 blocks) ~140s + fold; load dominates

# `tt-bio design --model <m>` for the single-shot designers. boltzgen is the one
# design model with a pipeline-stage progress stream of its own, so it keeps its
# own runner (run_gen) above; rfd3 and pxdesign share run_design and differ only in
# the fixture, the model-scoped flags and the per-design line the CLI prints.
# Each leg reuses a fixture already committed for that model's parity/perf work —
# no new fixture invented — and runs 1 design at a low step count: this gates UX
# plumbing, not accuracy or perf.
DESIGN_LEGS: dict[str, dict] = {
    "rfd3": dict(
        # The SAME IAI motif-scaffold fixture the parity gate and the perf leg use.
        spec=REPO_ROOT / "scripts" / "rfd3_port" / "parity_artifacts" / "iai_protein" / "iai_inputs.yaml",
        args=["--from_pdb", "--num_timesteps", "4"],   # 4 is the shipped CLI default
        pass_devices=True,
        # "  <spec_id>#<i>: <path>.cif (n atoms)"
        design_line=r"^\s*\S+#\d+:\s+\S+\.cif\s+\(\d+ atoms\)",
        timeout=1200,   # 0.65 GiB ckpt + first-kernel compile + 1 design; load dominates
    ),
    "pxdesign": dict(
        # PD-L1 (5O45 chain A, 116 residues) + an 80-residue binder — the fixture the
        # pxdesign input/write tests already use, self-contained (its structure file
        # resolves beside the YAML).
        spec=REPO_ROOT / "tests" / "fixtures" / "pxdesign" / "PDL1.yaml",
        args=["--n_step", "8"],   # shipped default is 400; 8 is a UX smoke
        pass_devices=False,   # pxdesign is batch-1 and reads its card from TT_VISIBLE_DEVICES
        # "  <stem>.cif: N residues, N atoms; target fit X A over N conditioned tokens"
        design_line=r"^\s*\S+\.cif:\s+\d+ residues,\s+\d+ atoms; target fit ",
        timeout=1200,   # 0.52 GiB ckpt + first-kernel compile + 8 steps
    ),
}

# `tt-bio affinity` — the scalar-output affinity predictors, which fold nothing:
# no coordinates, no CIF, so their UX is the stdout phases plus the two files the
# screen writes. Runs the SAME FKBP12+SB3 fixture as the boltz2-affinity leg and as
# both affinity cells in the perf gate, so all four surfaces score one target.
SCALAR_AFFINITY_MODELS = list(tt_bio_main.AFFINITY_MODELS)
SCALAR_AFFINITY_TIMEOUT_S = 900  # bf16 trunk, 5 recycles, 256-token crop at CLI defaults

# Checkpoint preconditions, as data rather than scattered ifs. Four models can be
# missing weights on a given host, and the right answer differs per model:
#
#   "require" — refuse to start and name the fix. rf3 and pxdesign auto-download,
#     so the fix is one command and a gate must not sit on a multi-GB fetch mid-leg.
#     openfold3's weights tt-bio cannot download, but RELEASING.md names OF3_CKPT a
#     release-host prerequisite, so a missing file there is a misconfigured gate host
#     rather than a legitimate absence — a green run with of3 ungated is worse than a
#     loud stop. Paths come from the weights registry, which honours OF3_CKPT and
#     TT_BIO_CACHE/TT_BIO_ROOT; a hardcoded ~/.boltz skipped the leg on any box with a
#     relocated cache while the weights sat right there.
#   "skip" — gate the leg when the checkpoint is present, SKIP it when absent with
#     the reason printed on its own row and again in the verdict line. openbind's
#     checkpoint is not a release-host prerequisite, and aborting all of the other
#     legs over one optional manual download makes the gate unrunnable on a fresh
#     box. A skip is loud and never counts as coverage.
CKPT_POLICY: dict[str, tuple[str, str]] = {
    "openfold3": ("require", "set OF3_CKPT or place the OpenFold3 preview2 weights at "
                             "{path} (see docs/openfold3-port.md)"),
    "openbind": ("skip", "OpenBind-0 checkpoint absent at {path} — tt-bio does not "
                         "download it (no published parameter licence). Fetch it per "
                         "docs/weights.md to gate this leg."),
    "rf3": ("require", "fetch it with `tt-bio weights --download rf3` or place it at {path}"),
    "pxdesign": ("require", "fetch it with `tt-bio weights --download pxdesign` or place "
                            "it at {path}"),
}

# Models behind a --model CLI choice that deliberately have NO UX leg. Empty on
# purpose: every shipped model is gated. An entry here needs a specific,
# mechanical reason — never "it shares a code path with a covered model", which is
# the exact reasoning that let opendde-abag ship with zero perf coverage.
LEGS_EXEMPT: dict[str, str] = {}

# esmc/saprot embed input: trpcage's 20-mer as a one-sequence FASTA, written into
# the per-run tmp dir so the gate is self-contained (no examples/FASTA dependency).
EMBED_SEQ = "NLYIQWLKDGGPSSGRPPPS"


def _subprocess_env(extra: dict | None = None) -> dict:
    """Environment for invoking ``tt_bio.main`` so it resolves to THIS worktree's
    tt_bio (PYTHONPATH=REPO_ROOT) regardless of any editable install pointing at
    another checkout. Matches the release_gate invocation convention."""
    env = dict(os.environ)
    pp = str(REPO_ROOT)
    existing = env.get("PYTHONPATH")
    if existing:
        pp = pp + os.pathsep + existing
    env["PYTHONPATH"] = pp
    if extra:
        env.update(extra)
    return env


def _run(cmd: list[str], *, env: dict | None = None, timeout: int | None = PER_MODEL_TIMEOUT_S,
         cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout+stderr. Raises TimeoutExpired on timeout.

    Defaults to ``PER_MODEL_TIMEOUT_S`` rather than ``None`` so a caller that
    forgets an explicit timeout still can't hang the gate forever on a wedged
    device or a flaky dependency (standing gate rule). Every current caller
    passes an explicit timeout, so this only hardens future ones."""
    return subprocess.run(cmd, cwd=str(cwd), env=env, timeout=timeout,
                          capture_output=True, text=True)


def _cli_predict(model: str, out_dir: Path, data: Path) -> list[str]:
    """Build the predict command for one fold model."""
    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(data),
        "--model", model,
        "--recycling_steps", str(RECYCLING_STEPS),
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--seed", str(SEED),
        "--out_dir", str(out_dir),
        "--debug",  # NullDisplay: clean headless, no Rich TTY animation
    ]
    if model in MSA_DEPENDENT:
        cmd.append("--single_sequence")
    return cmd


# ── leg 1: live progress view ──────────────────────────────────────────────

def _load_events(cap_path: Path) -> list[dict]:
    events = []
    for line in cap_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def _check_progress(events: list[dict]) -> list[str]:
    """Assert the event stream advances through trunk → diffusion → done with
    no phase skipped. Returns a list of problem strings (empty == pass)."""
    problems = []
    stages = [(e.get("stage"), e.get("step"), e.get("total"))
              for e in events if e.get("event") == "stage"]
    stage_names = [s[0] for s in stages]
    dones = [e for e in events if e.get("event") == "done"]

    if not events:
        return ["no progress events captured (TT_BIO_PROGRESS_CAPTURE not wired?)"]
    if not dones:
        problems.append("no 'done' event — predict did not report completion")
    elif not any(d.get("status") == "ok" for d in dones):
        problems.append(
            f"no 'done' event with status=ok (statuses: {[d.get('status') for d in dones]})")

    trunk = [s for s in stages if s[0] == "trunk"]
    diffusion = [s for s in stages if s[0] == "diffusion"]

    # The headline bug class: the trunk recycling phase is skipped, so the live
    # view jumps straight from loading/0 to diffusion.
    if not trunk:
        problems.append("trunk phase MISSING — the 0→diffusion / loading→diffusion "
                        "jump class of regression (no 'trunk' stage event at all)")
    elif not any((t[2] or 0) > 0 for t in trunk):
        problems.append(f"trunk phase present but total=0 on every tick — the "
                        f"'0 trunk iterations' bug: {trunk}")

    if not diffusion:
        problems.append("diffusion phase MISSING — no 'diffusion' stage event")

    if trunk and diffusion:
        ti = stage_names.index("trunk")
        di = stage_names.index("diffusion")
        if not ti < di:
            problems.append(f"trunk not before diffusion (trunk@{ti}, diffusion@{di}) "
                            f"— trunk phase is emitted after diffusion, so the live "
                            f"view would still jump past it")

    if stage_names and stage_names[0] == "diffusion":
        problems.append(f"first stage event is 'diffusion' — the loading→diffusion "
                        f"jump (first 4 stages: {stage_names[:4]})")

    # The per-phase ticks must advance monotonically — a regression that emits a
    # single end-of-phase event (no per-iteration / per-step ticking) would leave
    # steps flat or out of order.
    for name, evs in (("trunk", trunk), ("diffusion", diffusion)):
        steps = [e[1] for e in evs if e[1] is not None]
        if len(steps) >= 2 and steps != sorted(steps):
            problems.append(f"{name} steps not monotonic non-decreasing: {steps}")

    return problems


# ── leg 2: output files parse ──────────────────────────────────────────────

def _check_cif(cif: Path) -> list[str]:
    """Strict Bio.PDB.MMCIFParser parse — catches writer/format regressions."""
    try:
        from Bio.PDB import MMCIFParser
        from Bio.PDB.PDBExceptions import PDBConstructionWarning
    except ImportError:
        return ["biopython not installed (Bio.PDB.MMCIFParser unavailable)"]
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", PDBConstructionWarning)
            structure = MMCIFParser(QUIET=True).get_structure(NAME, str(cif))
        n_atoms = sum(1 for _ in structure.get_atoms())
        if n_atoms == 0:
            return [f"{cif.name}: parsed but contains 0 atoms"]
    except Exception as e:
        return [f"{cif.name}: CIF parse failed: {type(e).__name__}: {e}"]
    return []


def _check_npz(npz: Path, seq: str) -> list[str]:
    try:
        import numpy as np
    except ImportError:
        return ["numpy not installed"]
    try:
        z = np.load(npz, allow_pickle=False)
    except Exception as e:
        return [f"{npz.name}: npz load failed: {type(e).__name__}: {e}"]
    missing = [k for k in ("per_residue", "pooled", "sequence") if k not in z.files]
    if missing:
        return [f"{npz.name}: missing arrays {missing} (have {list(z.files)})"]
    try:
        pr = z["per_residue"]
        pooled = z["pooled"]
        loaded_seq = str(z["sequence"])
    except Exception as e:
        return [f"{npz.name}: array read failed: {e}"]
    if pr.ndim != 2:
        return [f"{npz.name}: per_residue ndim={pr.ndim}, expected 2"]
    if pooled.ndim != 1:
        return [f"{npz.name}: pooled ndim={pooled.ndim}, expected 1"]
    if loaded_seq != seq:
        return [f"{npz.name}: sequence mismatch (got len {len(loaded_seq)}, "
                f"expected {len(seq)})"]
    if pr.shape[0] != len(seq):
        return [f"{npz.name}: per_residue L={pr.shape[0]} != sequence len {len(seq)}"]
    return []


def _check_results_json(path: Path) -> list[str]:
    try:
        rows = json.loads(path.read_text())
    except Exception as e:
        return [f"results.json load failed: {type(e).__name__}: {e}"]
    if not isinstance(rows, list) or not rows:
        return [f"results.json is not a non-empty list (got {type(rows).__name__})"]
    ok = [r for r in rows if isinstance(r, dict) and r.get("status") == "ok"]
    if not ok:
        statuses = [r.get("status") for r in rows if isinstance(r, dict)]
        return [f"results.json has no ok row (statuses: {statuses})"]
    r = ok[0]
    missing = [k for k in ("id", "status") if k not in r]
    if missing:
        return [f"results.json ok row missing keys {missing}: {r}"]
    # Every fold surface writes a per-structure confidence metric the UI/CLI
    # summary reads — its absence is a real shape regression. boltz2 writes
    # complex_plddt / confidence_score; protenix-v2 / esmfold2 write plddt; all
    # write iptm/ptm. Accept any one — the point is a confidence number exists.
    confidence_keys = ("plddt", "complex_plddt", "complex_iplddt", "iptm",
                        "ptm", "confidence_score")
    if not any(k in r for k in confidence_keys):
        return [f"results.json ok row has no confidence metric (none of "
                f"{confidence_keys} present): {sorted(r)}"]
    return []


def _check_manifest(path: Path, seq_id: str, seq: str) -> list[str]:
    try:
        m = json.loads(path.read_text())
    except Exception as e:
        return [f"manifest.json load failed: {type(e).__name__}: {e}"]
    missing = [k for k in ("model", "pool", "format", "d_model", "dtype", "sequences")
               if k not in m]
    if missing:
        return [f"manifest.json missing keys {missing}: {sorted(m)}"]
    seqs = m["sequences"]
    if not any(s.get("id") == seq_id and s.get("length") == len(seq) for s in seqs):
        return [f"manifest.json sequences don't include {seq_id} L={len(seq)}: {seqs}"]
    return []


# ── leg 3: CLI behaves ─────────────────────────────────────────────────────

def _check_cli() -> list[str]:
    problems = []
    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "predict", "--help"],
                 env=_subprocess_env(), timeout=60)
    except Exception as e:
        return [f"predict --help failed to run: {e}"]
    if r.returncode != 0:
        problems.append(f"predict --help exited {r.returncode}")
    else:
        for flag in ("--model", "--sampling_steps", "--diffusion_samples",
                     "--recycling_steps", "--single_sequence", "--out_dir", "--seed"):
            if flag not in r.stdout:
                problems.append(f"predict --help missing flag {flag}")
        # Every shipped fold model must be listed as a choice a user can type. The
        # expected set is the CLI's own tuple, so a dropped choice fails here
        # instead of only surfacing when someone tries to run the model.
        for model_name in FOLD_MODELS:
            if model_name not in r.stdout:
                problems.append(f"predict --help does not list --model choice {model_name}")

    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "embed", "--help"],
                 env=_subprocess_env(), timeout=60)
    except Exception as e:
        problems.append(f"embed --help failed to run: {e}")
    else:
        if r.returncode != 0:
            problems.append(f"embed --help exited {r.returncode}")
        else:
            for flag in ("--model", "--format", "--out_dir", "--pool"):
                if flag not in r.stdout:
                    problems.append(f"embed --help missing flag {flag}")
            for model_name in tt_bio_main.EMBED_MODELS:
                if model_name not in r.stdout:
                    problems.append(f"embed --help does not list --model choice {model_name}")

    # SaProt ships under its own `tt-bio saprot` subcommand (not `tt-bio embed`),
    # so its flag surface is gated separately -- a regression that drops one of
    # its core flags would otherwise ship silently.
    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "saprot", "--help"],
                 env=_subprocess_env(), timeout=60)
    except Exception as e:
        problems.append(f"saprot --help failed to run: {e}")
    else:
        if r.returncode != 0:
            problems.append(f"saprot --help exited {r.returncode}")
        else:
            for flag in ("--model", "--format", "--out_dir", "--pool", "--structure"):
                if flag not in r.stdout:
                    problems.append(f"saprot --help missing flag {flag}")
            for model_name in tt_bio_main.SAPROT_MODELS:
                if model_name not in r.stdout:
                    problems.append(f"saprot --help does not list --model choice {model_name}")

    # `tt-bio gen` is the DEPRECATED hidden alias for `tt-bio design --model
    # boltzgen`: it must keep working (exit 0), print a one-line deprecation
    # warning to stderr, and forward --help to BoltzGen's own argparse parser
    # (tt_bio/boltzgen/cli/boltzgen.py), whose run help lists the core design
    # flags. It must also stay hidden from the top-level help.
    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "gen", "run", "--help"],
                 env=_subprocess_env(), timeout=60)
        if r.returncode != 0:
            problems.append(f"gen run --help exited {r.returncode}")
        else:
            if "deprecat" not in (r.stderr or "").lower():
                problems.append("gen run --help printed no deprecation warning on stderr")
            for flag in ("--num_designs", "--protocol", "--output", "--devices",
                         "--budget"):
                if flag not in r.stdout:
                    problems.append(f"gen run --help missing forwarded flag {flag}")
    except Exception as e:
        problems.append(f"gen run --help failed to run: {e}")
    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "--help"],
                 env=_subprocess_env(), timeout=60)
        if r.returncode != 0:
            problems.append(f"tt-bio --help exited {r.returncode}")
        elif re.search(r"^\s+gen\s", r.stdout, re.M):
            problems.append("deprecated `gen` alias is visible in tt-bio --help "
                            "(must stay hidden)")
    except Exception as e:
        problems.append(f"tt-bio --help failed to run: {e}")

    # `tt-bio design` is the unified design command with `--model
    # boltzgen|rfd3|pxdesign` (mirroring `predict --model`). Gate the shared flag
    # surface plus every model-scoped group so a regression that drops one ships
    # loudly.
    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "design", "--help"],
                 env=_subprocess_env(), timeout=60)
        if r.returncode != 0:
            problems.append(f"design --help exited {r.returncode}")
    except Exception as e:
        problems.append(f"design --help failed to run: {e}")
    else:
        for flag in ("--model", "--out_dir", "--num_designs", "--devices", "--seed",
                     "--from_pdb", "--num_timesteps", "--batch_size", "--checkpoint",
                     "--protocol", "--steps", "--budget", "--n_step"):
            if flag not in r.stdout:
                problems.append(f"design --help missing flag {flag}")
        for model_name in tt_bio_main.DESIGN_MODELS:
            if model_name not in r.stdout:
                problems.append(f"design --help does not mention --model choice {model_name}")
        if "--golden_dir" in r.stdout:
            problems.append("design --help still shows the deprecated --golden_dir flag "
                            "(must stay hidden)")

    # `tt-bio affinity` is its own verb (scalar affinity, no fold), so its flag
    # surface needs its own check — it had none, which is half of why nesso1 shipped
    # with no UX coverage.
    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "affinity", "--help"],
                 env=_subprocess_env(), timeout=60)
        if r.returncode != 0:
            problems.append(f"affinity --help exited {r.returncode}")
    except Exception as e:
        problems.append(f"affinity --help failed to run: {e}")
    else:
        for flag in ("--model", "--out_dir", "--accelerator", "--trunk",
                     "--recycling_steps", "--tokens_budget", "--devices", "--seed"):
            if flag not in r.stdout:
                problems.append(f"affinity --help missing flag {flag}")
        for model_name in tt_bio_main.AFFINITY_MODELS:
            if model_name not in r.stdout:
                problems.append(f"affinity --help does not list --model choice {model_name}")
    return problems


def _shipped_models() -> dict[str, tuple[str, ...]]:
    """Every ``*_MODELS`` tuple in ``tt_bio.main`` — the single source of truth each
    ``--model`` choice list is built from. DISCOVERED by name, not enumerated: written
    against the tuples that exist today, an enumerated version goes blind the moment a
    new CLI verb brings its own tuple, which is how `tt-bio affinity`'s nesso1 could
    ship with no coverage from a check built to make exactly that impossible."""
    return {n: getattr(tt_bio_main, n) for n in dir(tt_bio_main) if n.endswith("_MODELS")}


def _assert_full_model_coverage() -> None:
    """Fail loudly, before any device work, if a model shipped behind a ``--model``
    CLI choice has neither a UX leg nor a documented ``LEGS_EXEMPT`` reason.

    Three models shipped in 0.7.0 — nesso1, openbind, pxdesign — and all three had
    zero UX coverage, because the gated set was three hardcoded constants. A model
    absent from a hand-maintained list can stay uncovered forever and nothing goes
    red. This turns that silence into a startup failure that names the model.
    """
    tuples = _shipped_models()
    shipped = set().union(*tuples.values())
    covered = (set(FOLD_MODELS) | set(EMBED_MODELS) | {GEN_MODEL}
               | set(DESIGN_LEGS) | set(SCALAR_AFFINITY_MODELS))
    uncovered = shipped - covered - set(LEGS_EXEMPT)
    if uncovered:
        raise SystemExit(
            f"ux_regression.py: no UX leg and no LEGS_EXEMPT reason for "
            f"{sorted(uncovered)} — every model in tt_bio.main's "
            f"{', '.join(sorted(tuples))} must have a leg or an explicit reason. "
            f"A fold-shaped model needs nothing (FOLD_MODELS is derived); a design "
            f"model needs a DESIGN_LEGS entry or its own runner; a new CLI verb needs "
            f"a runner and its shape added to `covered` here.")


# ── per-model runners ──────────────────────────────────────────────────────

def run_fold(model: str, base: Path) -> dict:
    """Fold one model on its canonical tiny fixture, capture its progress stream,
    and gate the three UX legs. Returns a result row."""
    # opendde-abag is the antibody-antigen checkpoint and is gated on the Ab-Ag
    # fixture 1ahw_abag.yaml; every other fold model uses trpcage. The CLI path
    # is identical — only --model and the input file differ.
    data = ABAG_DATA if model == "opendde-abag" else DATA
    name = data.stem
    timeout = ABAG_MODEL_TIMEOUT_S if model == "opendde-abag" else PER_MODEL_TIMEOUT_S
    from tt_bio.main import predict_results_dir_name
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap_path = base / f"events_{model}.jsonl"
    cap_path.unlink(missing_ok=True)
    results_path = out_dir / predict_results_dir_name(model, name) / "results.json"
    struct_dir = out_dir / predict_results_dir_name(model, name) / "structures"

    env = _subprocess_env({"TT_BIO_PROGRESS_CAPTURE": str(cap_path)})

    cmd = _cli_predict(model, out_dir, data)
    print(f"\n{'='*70}\n[{model}] predict {data.name} (recyc={RECYCLING_STEPS}, "
          f"steps={SAMPLING_STEPS}, samples={DIFFUSION_SAMPLES})\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "progress": False, "parse": False,
           "results": False, "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        proc = _run(cmd, env=env, timeout=timeout)
    except subprocess.TimeoutExpired:
        row["error"] = f"predict timed out after {timeout}s"
        return row
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error"] = (f"predict exited {proc.returncode}: "
                        f"{tail[-1] if tail else ''}")
        return row

    # Leg 1: live progress view
    events = _load_events(cap_path) if cap_path.exists() else []
    prog_problems = _check_progress(events)
    row["checks"].append(f"progress: {'OK' if not prog_problems else 'FAIL'}")
    if prog_problems:
        row["checks"].extend(f"  • {p}" for p in prog_problems)
        if not row["error"]:
            row["error"] = "progress: " + "; ".join(prog_problems)

    # Leg 2: output CIF parses
    cifs = sorted(struct_dir.glob(f"{name}*.cif")) if struct_dir.exists() else []
    if not cifs:
        parse_problems = [f"predict wrote no CIF under {struct_dir}"]
    else:
        parse_problems = []
        for cif in cifs:
            parse_problems += _check_cif(cif)
    row["checks"].append(f"parse: {'OK' if not parse_problems else 'FAIL'}")
    if parse_problems:
        row["checks"].extend(f"  • {p}" for p in parse_problems)
        if not row["error"]:
            row["error"] = "parse: " + "; ".join(parse_problems)

    # Leg 3: results.json shape
    res_problems = _check_results_json(results_path) if results_path.exists() else [
        f"predict wrote no results.json at {results_path}"]

    row["progress"] = not prog_problems
    row["parse"] = not parse_problems
    row["results"] = not res_problems
    row["gate"] = row["progress"] and row["parse"] and row["results"]
    if res_problems:
        row["checks"].append(f"results.json: {'OK' if not res_problems else 'FAIL'}")
        row["checks"].extend(f"  • {p}" for p in res_problems)
        if not row["error"]:
            row["error"] = "results.json: " + "; ".join(res_problems)
    else:
        row["checks"].append("results.json: OK")
    return row


def _embed_subcommand(model: str) -> str:
    """The tt-bio CLI subcommand an embed model ships under. esmc-* use `embed`;
    saprot-* use their own `saprot` subcommand (SaProt has its own CLI entry, not
    the esmc embed command). Both accept the same --model/--out_dir/--format/--pool
    flags and write the same npz + manifest.json shape."""
    return "saprot" if model.startswith("saprot") else "embed"


def run_embed(model: str, base: Path) -> dict:
    """Run an embed model (esmc-* via `tt-bio embed`, saprot-* via `tt-bio saprot`)
    on a tiny sequence and gate the UX legs (embed has no fold phases — its
    user-facing progress is the load → embed → done stdout lines)."""
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_id = "tiny"
    fasta = out_dir / "tiny.fasta"
    fasta.write_text(f">{seq_id}\n{EMBED_SEQ}\n")

    cmd = [
        sys.executable, "-m", "tt_bio.main", _embed_subcommand(model), str(fasta),
        "--model", model, "--out_dir", str(out_dir), "--format", "npz",
    ]
    print(f"\n{'='*70}\n[{model}] embed {seq_id} (L={len(EMBED_SEQ)})\n{'='*70}",
          flush=True)

    row = {"model": model, "seconds": None, "progress": False, "parse": False,
           "manifest": False, "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        proc = _run(cmd, env=_subprocess_env(), timeout=PER_MODEL_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        row["error"] = f"embed timed out after {PER_MODEL_TIMEOUT_S}s"
        return row
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error"] = (f"embed exited {proc.returncode}: "
                        f"{tail[-1] if tail else ''}")
        return row

    # Leg 1 (embed): the user-facing load → embed → done stdout lines, in order.
    lines = [l for l in (proc.stdout or "").splitlines() if l.strip()]
    lower = [l.lower() for l in lines]
    prog_problems = []
    li = next((i for i, l in enumerate(lower) if "loading" in l), None)
    ei = next((i for i, l in enumerate(lower) if "embedding" in l), None)
    di = next((i for i, l in enumerate(lower) if l.startswith("done") or " — " in l and "done" in l), None)
    if li is None or ei is None or di is None:
        prog_problems.append(f"missing load→embed→done stdout lines "
                             f"(loading@{li}, embedding@{ei}, done@{di})")
    elif not (li < ei < di):
        prog_problems.append(f"stdout phases out of order: loading@{li}, "
                             f"embedding@{ei}, done@{di}")
    row["checks"].append(f"progress(stdout): {'OK' if not prog_problems else 'FAIL'}")
    if prog_problems:
        row["checks"].extend(f"  • {p}" for p in prog_problems)

    # Leg 2: npz parses with the expected shape.
    npz = out_dir / f"{seq_id}.npz"
    parse_problems = _check_npz(npz, EMBED_SEQ) if npz.exists() else [
        f"embed wrote no npz at {npz}"]
    row["checks"].append(f"parse(npz): {'OK' if not parse_problems else 'FAIL'}")
    if parse_problems:
        row["checks"].extend(f"  • {p}" for p in parse_problems)

    # Leg 3: manifest.json shape.
    manifest = out_dir / "manifest.json"
    man_problems = _check_manifest(manifest, seq_id, EMBED_SEQ) if manifest.exists() else [
        f"embed wrote no manifest.json at {manifest}"]
    row["checks"].append(f"manifest: {'OK' if not man_problems else 'FAIL'}")
    if man_problems:
        row["checks"].extend(f"  • {p}" for p in man_problems)

    row["progress"] = not prog_problems
    row["parse"] = not parse_problems
    row["manifest"] = not man_problems
    row["gate"] = row["progress"] and row["parse"] and row["manifest"]
    if not row["error"]:
        for p in (prog_problems + parse_problems + man_problems):
            row["error"] = (row["error"] + "; " if row["error"] else "") + p
    return row


# ── boltzgen (binder design) ───────────────────────────────────────────────

# The gen pipeline's own progress reporter (tt_bio/boltzgen/progress.py) emits
# plain-text stage events on stdout under `--debug --log` (DebugReporter):
#   >>> [idx/total] <step_name>      stage start
#       <label> <n>/<total>          sub-step tick (trunk / diff / batch / msa)
#   <<< ✓                            stage done
# This is the headless equivalent of the fold leg's JSONL event stream — same
# real pipeline stages, not a scraped TTY or synthetic replay.
_GEN_STAGE_START = ">>> "   # DebugReporter.stage_start prefix
_GEN_STAGE_DONE = "<<< "    # DebugReporter.stage_done prefix


def _check_gen_progress(stdout: str) -> list[str]:
    """Assert the gen pipeline's stdout stage stream advances through the
    design + refold + analysis stages with no phase skipped. Returns problem
    strings (empty == pass)."""
    problems = []
    # A sub-step tick: "    <label> <n>/<total>" (DebugReporter.step) where label
    # is one of trunk/diff/batch/msa. Match on the stripped line.
    _tick = re.compile(r"^(trunk|diff|batch|msa)\s+\d+/\d+$")
    starts: list[tuple[int, str]] = []
    dones = 0
    steps = 0
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith(_GEN_STAGE_START) and "/" in s:
            # ">>> [idx/total] step_name"
            tail = s[len(_GEN_STAGE_START):]
            try:
                name = tail.split("]", 1)[1].strip()
            except IndexError:
                name = ""
            starts.append((len(starts) + 1, name))
        elif s.startswith(_GEN_STAGE_DONE):
            if "✓" in s:
                dones += 1
        elif _tick.match(s):
            steps += 1

    if not starts:
        return ["no `>>> [i/N] <step>` stage-start lines captured "
                "(design --debug --log progress not wired?)"]
    names = [n for _, n in starts]
    # protein-anything runs: design → inverse_folding → folding → design_folding
    # → analysis → filtering (design_folding is the isolated refold = the
    # designability metric's source). Require the headline design + refold +
    # analysis stages so a regression that skips or reorders a phase fails.
    for required in ("design", "analysis"):
        if required not in names:
            problems.append(f"'{required}' stage MISSING from design progress "
                            f"(stages seen: {names})")
    if "design_folding" not in names and "folding" not in names:
        problems.append("no refold stage (design_folding/folding) — the isolated "
                        f"refold phase is missing (stages seen: {names})")
    if dones == 0:
        problems.append("no `<<< ✓` stage-done lines — pipeline did not report "
                        "any completed stage")
    if steps == 0:
        problems.append("no sub-step tick lines (trunk/diff/batch/msa) — the "
                        "per-stage progress did not tick")
    # Stages must advance in declaration order; a reordering would surface as a
    # duplicate or out-of-order name sequence.
    if len(names) != len(set(names)):
        problems.append(f"stage names repeat (out-of-order emission): {names}")
    return problems


def run_gen(model: str, base: Path) -> dict:
    """Run one tiny ``tt-bio design --model boltzgen`` binder-design job and gate
    the three UX legs (progress phases, output parses, results shape). Returns a
    result row."""
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not GEN_SPEC.exists():
        sys.exit(f"missing gen fixture {GEN_SPEC}")

    # --devices takes physical card ids (not a count) — same convention as the
    # rfd3 design leg below.
    visible = (os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
    cmd = [
        sys.executable, "-m", "tt_bio.main", "design", str(GEN_SPEC),
        "--model", "boltzgen",
        "--out_dir", str(out_dir),
        "--num_designs", str(GEN_NUM_DESIGNS),
        "--protocol", GEN_PROTOCOL,
        "--devices", visible,
        "--budget", str(GEN_NUM_DESIGNS),
        "--debug", "--log",   # DebugReporter: plain-text stage events on stdout
    ]
    print(f"\n{'='*70}\n[{model}] design --model boltzgen {GEN_SPEC.name} "
          f"({GEN_PROTOCOL}, {GEN_NUM_DESIGNS} design)\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "progress": False, "parse": False,
           "metrics": False, "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        proc = _run(cmd, env=_subprocess_env(), timeout=GEN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        row["error"] = f"design run timed out after {GEN_TIMEOUT_S}s"
        return row
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error"] = (f"design run exited {proc.returncode}: "
                        f"{tail[-1] if tail else ''}")
        return row

    # Leg 1: live progress view (gen's stdout stage stream).
    prog_problems = _check_gen_progress(proc.stdout or "")
    row["checks"].append(f"progress: {'OK' if not prog_problems else 'FAIL'}")
    if prog_problems:
        row["checks"].extend(f"  • {p}" for p in prog_problems)
        if not row["error"]:
            row["error"] = "progress: " + "; ".join(prog_problems)

    # Leg 2: written CIFs parse under a strict standard parser.
    cifs = sorted(out_dir.rglob("*.cif")) if out_dir.exists() else []
    if not cifs:
        parse_problems = [f"design run wrote no CIF under {out_dir}"]
    else:
        parse_problems = []
        for cif in cifs:
            parse_problems += _check_cif(cif)
    row["checks"].append(f"parse: {'OK' if not parse_problems else 'FAIL'}")
    if parse_problems:
        row["checks"].extend(f"  • {p}" for p in parse_problems)
        if not row["error"]:
            row["error"] = "parse: " + "; ".join(parse_problems)

    # Leg 3: the analysis metrics table the designability harness reads exists
    # and has the designability RMSD column (the user-facing QA output).
    metrics_problems = _check_gen_metrics(out_dir)
    row["checks"].append(f"metrics: {'OK' if not metrics_problems else 'FAIL'}")
    if metrics_problems:
        row["checks"].extend(f"  • {p}" for p in metrics_problems)
        if not row["error"]:
            row["error"] = "metrics: " + "; ".join(metrics_problems)

    row["progress"] = not prog_problems
    row["parse"] = not parse_problems
    row["metrics"] = not metrics_problems
    row["gate"] = row["progress"] and row["parse"] and row["metrics"]
    return row


def _check_gen_metrics(out_dir: Path) -> list[str]:
    """The gen pipeline's analysis step writes aggregate_metrics_*.csv with a
    designability RMSD column (the same column the accuracy leg harvests). Its
    absence is a real shape regression in the user-facing QA output."""
    try:
        import csv as _csv
    except ImportError:
        return ["csv module unavailable"]
    hits = sorted(out_dir.rglob("aggregate_metrics_*.csv"))
    if not hits:
        return [f"no aggregate_metrics_*.csv under {out_dir} — analysis did not run"]
    csv_path = min(hits, key=lambda p: len(p.parts))  # merged top-level table
    try:
        with open(csv_path, newline="") as fh:
            rows = list(_csv.DictReader(fh))
    except Exception as e:
        return [f"{csv_path.name}: read failed: {e}"]
    if not rows:
        return [f"{csv_path.name}: empty metrics table"]
    cols = set(rows[0].keys())
    # Match the designability harness's SC_COLUMNS preference order.
    if "designfolding-bb_rmsd" not in cols and "bb_rmsd_design" not in cols:
        return [f"{csv_path.name}: no designability RMSD column "
                f"(have {sorted(cols)})"]
    return []


def _check_design_progress(stdout: str, design_line: str) -> list[str]:
    """Assert `tt-bio design`'s stdout advances Designing -> per-design -> Done. The
    design command's progress is plain print (no Rich live view, no stage stream like
    gen), so the phases ARE these lines, and their order is what a user watches.
    ``design_line`` is the model's own per-design result line. Returns problem strings
    (empty == pass)."""
    problems = []
    lines = stdout.splitlines()
    di = next((i for i, l in enumerate(lines) if "Designing" in l), None)
    ri = next((i for i, l in enumerate(lines) if re.match(design_line, l)), None)
    fi = next((i for i, l in enumerate(lines) if "Done —" in l or "Done -" in l), None)
    if di is None:
        problems.append("no 'Designing ...' headline — design start not reported")
    if ri is None:
        problems.append(f"no per-design result line matching {design_line!r} — the design "
                        f"itself is never reported, so the user sees start then finish "
                        f"with nothing in between")
    if fi is None:
        problems.append("no 'Done — ...' line — design completion not reported")
    # Both design models print the summary before the per-design detail, so the
    # invariant worth asserting is that the headline comes first: a run that reports
    # a result or a completion before it reports starting is the regression.
    for name, idx in (("Done", fi), ("the per-design result", ri)):
        if di is not None and idx is not None and idx < di:
            problems.append(f"{name} line is printed before the 'Designing ...' headline "
                            f"(Designing@{di}, {name}@{idx})")
    return problems


def run_design(model: str, base: Path) -> dict:
    """Run one tiny single-shot `tt-bio design --model <m>` job and gate the UX legs
    (progress lines in order, output CIF parses). Table-driven off DESIGN_LEGS, so
    rfd3 and pxdesign share this runner and a third design model needs a fixture and
    a per-design line pattern, not another copy of this function."""
    leg = DESIGN_LEGS[model]
    spec, timeout = leg["spec"], leg["timeout"]
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not spec.exists():
        sys.exit(f"missing design fixture {spec}")

    cmd = [
        sys.executable, "-m", "tt_bio.main", "design", str(spec),
        "--model", model,
        "--out_dir", str(out_dir),
        "--num_designs", "1",
        *leg["args"],
    ]
    if leg["pass_devices"]:
        # --devices takes physical card ids (not a count); a hardcoded "1" fails on
        # single-card hosts (pc has only id 0). Derive from TT_VISIBLE_DEVICES
        # (default 0) so the leg runs on the caller's pinned card.
        visible = (os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0].strip() or "0")
        cmd += ["--devices", visible]
    print(f"\n{'='*70}\n[{model}] design {spec.name} "
          f"(1 design, {' '.join(leg['args'])})\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "progress": False, "parse": False,
           "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        proc = _run(cmd, env=_subprocess_env(), timeout=timeout)
    except subprocess.TimeoutExpired:
        row["error"] = f"design timed out after {timeout}s"
        return row
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error"] = (f"design exited {proc.returncode}: "
                        f"{tail[-1] if tail else ''}")
        return row

    # Leg 1: progress lines (design's plain-print Designing -> per-design -> Done).
    prog_problems = _check_design_progress(proc.stdout or "", leg["design_line"])
    row["checks"].append(f"progress: {'OK' if not prog_problems else 'FAIL'}")
    if prog_problems:
        row["checks"].extend(f"  • {p}" for p in prog_problems)
        if not row["error"]:
            row["error"] = "progress: " + "; ".join(prog_problems)

    # Leg 2: written CIF parses under a strict standard parser.
    cifs = sorted(out_dir.rglob("*.cif")) if out_dir.exists() else []
    if not cifs:
        parse_problems = [f"design wrote no CIF under {out_dir}"]
    else:
        parse_problems = []
        for cif in cifs:
            parse_problems += _check_cif(cif)
    row["checks"].append(f"parse: {'OK' if not parse_problems else 'FAIL'}")
    if parse_problems:
        row["checks"].extend(f"  • {p}" for p in parse_problems)
        if not row["error"]:
            row["error"] = "parse: " + "; ".join(parse_problems)

    row["progress"] = not prog_problems
    row["parse"] = not parse_problems
    # design has no separate results.json/metrics table like gen — the CIF +
    # progress lines ARE the user-facing output, so gate = progress & parse.
    row["gate"] = row["progress"] and row["parse"]
    return row


# ── driver ─────────────────────────────────────────────────────────────────

# Every leg row carries the same keys plus its own shape-specific booleans, so one
# printer covers all of them — and unlike the five it replaces, it prints the
# per-check detail for every shape instead of dropping it for design/affinity.
_ROW_FLAGS = ("progress", "parse", "results", "manifest", "metrics")


def _print_row(r: dict) -> None:
    wall = f"{r['seconds']:.0f}s" if r.get("seconds") is not None else "-"
    if r.get("skipped"):
        print(f"{r['model']:<16}{wall:>9}  SKIP — {r['reason']}")
        return
    verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r.get("error") else "FAIL"
    print(f"{r['model']:<16}{wall:>9}  {verdict}")
    print("  " + " ".join(f"{k}={r[k]}" for k in _ROW_FLAGS if k in r))
    for c in r.get("checks", ()):
        print(f"  {c}")


def _check_affinity_results(path: Path) -> list[str]:
    """Affinity results.json shape: the fold leg's confidence metric AND the
    user-facing affinity scalar. The affinity_pred_value (MW-corrected
    log10(IC50)) is the whole point of affinity mode — its absence is a real
    shape regression in the customer-facing output."""
    problems = _check_results_json(path)
    try:
        rows = json.loads(path.read_text())
    except Exception as e:
        return problems + [f"results.json load failed: {type(e).__name__}: {e}"]
    ok = [r for r in rows if isinstance(r, dict) and r.get("status") == "ok"]
    if not ok:
        return problems
    r = ok[0]
    if "affinity_pred_value" not in r:
        problems.append(f"results.json ok row has no affinity_pred_value "
                        f"(affinity mode did not emit the user-facing scalar): "
                        f"{sorted(r)}")
    return problems


def run_affinity(model: str, base: Path) -> dict:
    """Run one tiny ``tt-bio predict`` affinity-mode call (FKBP12+SB3) and gate the
    three UX legs: live progress phases advance correctly on the affinity path
    (the must-never-recur jump class), the written CIF parses, and results.json
    carries the user-facing affinity scalar. Returns a result row."""
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not AFFINITY_SPEC.exists():
        sys.exit(f"missing affinity fixture {AFFINITY_SPEC}")
    from tt_bio.main import predict_results_dir_name
    cap_path = base / f"events_{model}.jsonl"
    cap_path.unlink(missing_ok=True)
    results_path = out_dir / predict_results_dir_name("boltz2", AFFINITY_SPEC.stem) / "results.json"
    struct_dir = out_dir / predict_results_dir_name("boltz2", AFFINITY_SPEC.stem) / "structures"

    env = _subprocess_env({"TT_BIO_PROGRESS_CAPTURE": str(cap_path)})
    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(AFFINITY_SPEC),
        "--model", "boltz2",
        "--single_sequence",
        "--override",
        "--affinity_mw_correction",
        "--debug",  # NullDisplay: clean headless, no Rich TTY
        "--recycling_steps", str(RECYCLING_STEPS),
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--sampling_steps_affinity", str(SAMPLING_STEPS),
        "--diffusion_samples_affinity", str(DIFFUSION_SAMPLES),
        "--out_dir", str(out_dir),
    ]
    print(f"\n{'='*70}\n[{model}] predict {AFFINITY_SPEC.name} (affinity mode, "
          f"recyc={RECYCLING_STEPS}, steps={SAMPLING_STEPS}, samples={DIFFUSION_SAMPLES})"
          f"\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "progress": False, "parse": False,
           "results": False, "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        proc = _run(cmd, env=env, timeout=AFFINITY_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        row["error"] = f"affinity predict timed out after {AFFINITY_TIMEOUT_S}s"
        return row
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error"] = (f"affinity predict exited {proc.returncode}: "
                        f"{tail[-1] if tail else ''}")
        return row

    # Leg 1: live progress view — same shape as a structure fold (verified on a
    # real affinity run: the affinity model's own trunk+diffusion re-run is
    # silent). _check_progress asserts trunk → diffusion → done with no phase
    # skipped — the exact jump-class guard GOALS.md calls out.
    events = _load_events(cap_path) if cap_path.exists() else []
    prog_problems = _check_progress(events)
    row["checks"].append(f"progress: {'OK' if not prog_problems else 'FAIL'}")
    if prog_problems:
        row["checks"].extend(f"  • {p}" for p in prog_problems)
        if not row["error"]:
            row["error"] = "progress: " + "; ".join(prog_problems)

    # Leg 2: written CIF parses under a strict standard parser.
    cifs = sorted(struct_dir.glob(f"{AFFINITY_SPEC.stem}*.cif")) if struct_dir.exists() else []
    if not cifs:
        parse_problems = [f"predict wrote no CIF under {struct_dir}"]
    else:
        parse_problems = []
        for cif in cifs:
            parse_problems += _check_cif(cif)
    row["checks"].append(f"parse: {'OK' if not parse_problems else 'FAIL'}")
    if parse_problems:
        row["checks"].extend(f"  • {p}" for p in parse_problems)
        if not row["error"]:
            row["error"] = "parse: " + "; ".join(parse_problems)

    # Leg 3: results.json shape — fold confidence metric AND affinity scalar.
    res_problems = _check_affinity_results(results_path) if results_path.exists() else [
        f"predict wrote no results.json at {results_path}"]
    row["checks"].append(f"results.json: {'OK' if not res_problems else 'FAIL'}")
    if res_problems:
        row["checks"].extend(f"  • {p}" for p in res_problems)
        if not row["error"]:
            row["error"] = "results.json: " + "; ".join(res_problems)

    row["progress"] = not prog_problems
    row["parse"] = not parse_problems
    row["results"] = not res_problems
    row["gate"] = row["progress"] and row["parse"] and row["results"]
    return row


# ── nesso1 (`tt-bio affinity`) ─────────────────────────────────────────────

def _check_scalar_affinity_progress(stdout: str) -> list[str]:
    """`tt-bio affinity` has no Rich live view and no event stream: its three plain
    stdout phases ARE the progress the user watches — `Loading <model> …`, one scored
    line per input, then `Done — n/n scored`. A screen that printed nothing until the
    end is the same class of regression as a fold bar jumping past a phase."""
    lines = [l.strip() for l in stdout.splitlines() if l.strip()]
    problems = []
    li = next((i for i, l in enumerate(lines) if l.lower().startswith("loading")), None)
    si = next((i for i, l in enumerate(lines)
               if re.match(r"^\S+: affinity -?[\d.]+ p\(binder\) [\d.]+ \(\d+ tokens", l)),
              None)
    di = next((i for i, l in enumerate(lines) if l.lower().startswith("done")), None)
    if li is None or si is None or di is None:
        problems.append(f"missing loading → per-input → done stdout lines "
                        f"(loading@{li}, scored@{si}, done@{di})")
    elif not li < si < di:
        problems.append(f"stdout phases out of order: loading@{li}, scored@{si}, "
                        f"done@{di}")
    return problems


def _check_scalar_affinity_results(out_dir: Path, stem: str) -> list[str]:
    """The two files a screen writes: `<id>_affinity.json` per input and one
    `affinity.csv` for the screen. The affinity scalar is the whole output of the
    model, so a row without it is a shape regression in the only thing the user gets."""
    problems = []
    js = out_dir / f"{stem}_affinity.json"
    if not js.exists():
        problems.append(f"affinity wrote no {js.name} under {out_dir}")
    else:
        try:
            row = json.loads(js.read_text())
        except Exception as e:
            problems.append(f"{js.name}: load failed: {type(e).__name__}: {e}")
        else:
            missing = [k for k in ("id", "n_tokens", "seconds", "affinity_pred_value",
                                   "affinity_probability_binary") if k not in row]
            if missing:
                problems.append(f"{js.name}: missing keys {missing} (have {sorted(row)})")
    csv_path = out_dir / "affinity.csv"
    if not csv_path.exists():
        return problems + [f"affinity wrote no affinity.csv at {csv_path}"]
    import csv as _csv
    try:
        with csv_path.open(newline="") as fh:
            rows = list(_csv.DictReader(fh))
    except Exception as e:
        return problems + [f"affinity.csv: read failed: {type(e).__name__}: {e}"]
    if not rows:
        return problems + ["affinity.csv has a header but no data row"]
    r = rows[0]
    if r.get("error"):
        problems.append(f"affinity.csv row errored: {r['error']}")
    if not r.get("affinity_pred_value"):
        problems.append(f"affinity.csv row has no affinity_pred_value "
                        f"(columns {sorted(r)})")
    return problems


def run_scalar_affinity(model: str, base: Path) -> dict:
    """Run one `tt-bio affinity` screen and gate its UX legs: the stdout phases
    advance in order, and both written files carry the affinity scalar. Everything is
    left at the shipped CLI defaults (bf16 trunk, 5 recycles, 256-token crop) so the
    leg exercises what a user actually gets."""
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not AFFINITY_SPEC.exists():
        sys.exit(f"missing affinity fixture {AFFINITY_SPEC}")
    cmd = [
        sys.executable, "-m", "tt_bio.main", "affinity", str(AFFINITY_SPEC),
        "--model", model,
        "--out_dir", str(out_dir),
    ]
    print(f"\n{'='*70}\n[{model}] affinity {AFFINITY_SPEC.name} (CLI defaults)"
          f"\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "progress": False, "results": False,
           "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        proc = _run(cmd, env=_subprocess_env(), timeout=SCALAR_AFFINITY_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        row["error"] = f"affinity timed out after {SCALAR_AFFINITY_TIMEOUT_S}s"
        return row
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error"] = f"affinity exited {proc.returncode}: {tail[-1] if tail else ''}"
        return row

    prog_problems = _check_scalar_affinity_progress(proc.stdout or "")
    row["checks"].append(f"progress(stdout): {'OK' if not prog_problems else 'FAIL'}")
    row["checks"].extend(f"  • {x}" for x in prog_problems)
    res_problems = _check_scalar_affinity_results(out_dir, AFFINITY_SPEC.stem)
    row["checks"].append(f"affinity.csv + json: {'OK' if not res_problems else 'FAIL'}")
    row["checks"].extend(f"  • {x}" for x in res_problems)

    row["progress"] = not prog_problems
    row["results"] = not res_problems
    row["gate"] = row["progress"] and row["results"]
    if not row["gate"]:
        row["error"] = "; ".join(prog_problems + res_problems)
    return row


def _ckpt_gate(model: str) -> tuple[str, str] | None:
    """``(action, message)`` when *model*'s checkpoint is missing, else None. Action
    is CKPT_POLICY's: "require" (refuse to start) or "skip" (skip the leg loudly)."""
    if model not in CKPT_POLICY:
        return None
    path = weights.resolve(model)
    if path is not None and path.exists():
        return None
    action, msg = CKPT_POLICY[model]
    return action, msg.format(path=path)


# Which runner drives which model, in run order. One table, so main() dispatches by
# lookup instead of one hand-written loop per shape, and adding a leg is one row.
RUNNERS: tuple[tuple, ...] = (
    (run_fold, FOLD_MODELS),
    (run_embed, EMBED_MODELS),
    (run_gen, [GEN_MODEL]),
    (run_affinity, [AFFINITY_MODEL]),
    (run_scalar_affinity, SCALAR_AFFINITY_MODELS),
    (run_design, list(DESIGN_LEGS)),
)
ALL_LEGS = [m for _, group in RUNNERS for m in group]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _assert_full_model_coverage()
    ap.add_argument("--model", action="append", choices=ALL_LEGS,
                    help="Gate only this model (repeatable). Default: every model behind "
                         "a --model CLI choice, plus Boltz-2's affinity mode.")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the per-run output dirs under the tmp dir for inspection.")
    ap.add_argument("--cli-only", action="store_true",
                    help="Run ONLY the CLI-behaviour leg (predict/embed --help). No card "
                         "needed — usable in GitHub CI. Skips the on-device legs.")
    args = ap.parse_args()

    # rf3's UX leg reported FAIL on 2026-08-23 because the gate host's env was missing
    # `toolz`, declared in pyproject.toml the day before. Every fixture and checkpoint was
    # present; the interpreter was not, and nothing said so.
    dep_problems = gate_guard.declared_dependency_problems(REPO_ROOT / "pyproject.toml")
    if dep_problems:
        for problem in dep_problems:
            print(f"PREFLIGHT - {problem}")
        sys.exit("Refusing to score tt-bio on an interpreter that does not satisfy its own "
                 "declared dependencies: a leg that dies on a missing import is reported as a "
                 "product failure.")

    # The guard drives the real `tt_bio.main` CLI via sys.executable, so it must
    # be launched with a Python that has tt-bio's deps installed (numpy / ttnn /
    # biopython) — i.e. the project venv, exactly like scripts/release_gate.py:
    #     /path/to/env/bin/python scripts/ux_regression.py
    # PYTHONPATH=REPO_ROOT (set by _subprocess_env) makes tt_bio resolve to this
    # worktree, so an editable install pointing at another checkout can't shadow it.
    probe = _run([sys.executable, "-c", "import tt_bio"],
                 env=_subprocess_env(), timeout=60)
    if probe.returncode != 0:
        sys.exit(
            f"this Python ({sys.executable}) cannot import tt_bio with "
            f"PYTHONPATH={REPO_ROOT}:\n{(probe.stderr or probe.stdout).strip()}\n"
            f"Run the guard with the project venv, e.g. "
            f"/home/ttuser/tt-bio-dev/env/bin/python scripts/ux_regression.py")

    # Leg 3 (CLI behaves) runs always — it needs no card.
    print(f"\n{'#'*78}\nUX GATE — leg 3: CLI behaves (predict / embed / saprot / "
          f"affinity / design / deprecated gen alias)\n{'#'*78}")
    cli_problems = _check_cli()
    all_pass = not cli_problems
    if cli_problems:
        for prob in cli_problems:
            print(f"  ✗ {prob}")
    else:
        n_shipped = len(set().union(*_shipped_models().values()))
        print(f"  ✓ predict / embed / saprot / affinity / design --help, deprecated "
              f"gen run --help (warns), tt-bio --help: all exit 0, list their core "
              f"flags, and list all {n_shipped} shipped --model choices")
    print(f"{'#'*78}")

    if args.cli_only:
        return 0 if all_pass else 1

    models = args.model or ALL_LEGS
    requested = [(runner, m) for runner, group in RUNNERS for m in group if m in models]

    if not requested:
        return 0 if all_pass else 1
    if not DATA.exists() and any(m in FOLD_MODELS for _, m in requested):
        sys.exit(f"missing gate target {DATA}")
    # Checkpoint preconditions, before the tmp dir and before any device work.
    # "require" stops the gate and names the fix; "skip" is recorded and printed.
    skips: dict[str, str] = {}
    for _, m in requested:
        gate = _ckpt_gate(m)
        if gate is None:
            continue
        action, message = gate
        if action == "require":
            sys.exit(f"missing {m} checkpoint: {message}. Refusing to skip: {m} is a "
                     f"shipped --model choice, and skipping it is how a model reaches a "
                     f"release with no UX coverage at all. Run the rest with --model <name>.")
        skips[m] = message

    base = Path(tempfile.mkdtemp(prefix="ux_gate_", dir=str(REPO_ROOT)))
    try:
        rows = []
        for runner, m in requested:
            if m in skips:
                rows.append({"model": m, "skipped": True, "reason": skips[m]})
                print(f"\n{'='*70}\n[{m}] SKIP — {skips[m]}\n{'='*70}", flush=True)
                continue
            r = runner(m, base)
            rows.append(r)
            all_pass &= r["gate"]
        print(f"\n{'#'*78}\nUX GATE — summary (fold fixtures: {DATA.name}"
              f"{f' / {ABAG_DATA.name} (opendde-abag)' if ABAG_DATA.exists() else ''}, "
              f"recyc={RECYCLING_STEPS}, steps={SAMPLING_STEPS}, "
              f"samples={DIFFUSION_SAMPLES}, seed={SEED})\n{'#'*78}")
        for r in rows:
            _print_row(r)
        print(f"{'#'*78}")
        skipped = [r["model"] for r in rows if r.get("skipped")]
        note = (f" — {len(skipped)} leg(s) NOT gated on this host: {', '.join(skipped)}"
                if skipped else "")
        print(f"GATE PASS — every surface run cleared progress + parse + "
              f"results/manifest shape, and the CLI behaves{note}" if all_pass
              else "GATE FAIL — a surface missed a UX leg (see above). A UX regression "
                   f"blocks a tag, same standing as an accuracy regression.{note}")
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
