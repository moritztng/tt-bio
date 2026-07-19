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
  3. CLI behaves — ``tt-bio predict --help`` / ``tt-bio embed --help`` exit 0
     and list the core flags, and each surface's results/manifest file has the
     shape the downstream reader expects.

Coverage: the six fold models (boltz2, esmfold2, esmfold2-fast, protenix-v2,
opendde, opendde-abag) for legs 1–3, plus esmc-600m embed for legs 2–3 (embed has
no fold phases; its user-facing progress is the load → embed → done stdout lines),
plus boltzgen for legs 1–3 exercised via `tt-bio gen run` (a tiny 1-design
binder job on examples/binder.yaml; its progress is the gen pipeline's own
stdout stage stream under --debug --log). opendde-abag is gated on the Ab-Ag
fixture examples/1ahw_abag.yaml; the other fold models use examples/trpcage.yaml.

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
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# trpcage (20 residues) is the canonical tiny fold target — small enough that
# even the ESMC-6B ESMFold2 load dominates wall-clock, so the gate stays fast.
DATA = REPO_ROOT / "examples" / "trpcage.yaml"
NAME = DATA.stem  # "trpcage" -> predict writes boltz_results_trpcage/

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

FOLD_MODELS = ["boltz2", "esmfold2", "esmfold2-fast", "protenix-v2", "opendde",
               "opendde-abag"]
# MSA-dependent models get --single_sequence so the gate is offline + deterministic
# (no ColabFold server round-trip). esmfold2 / esmfold2-fast are single-seq by design.
# opendde-abag rides the same MSA-dependent path as opendde (only the checkpoint
# differs — opendde_abag.pt vs opendde.pt), so it gets --single_sequence too.
MSA_DEPENDENT = {"boltz2", "protenix-v2", "opendde", "opendde-abag"}
# opendde-abag is the antibody-antigen checkpoint, so it is gated on the canonical
# Ab-Ag fixture 1ahw_abag.yaml (the same SAbDab/PDB 1ahw target the benchmark uses
# elsewhere) instead of trpcage. Every other fold model uses trpcage.
ABAG_DATA = REPO_ROOT / "examples" / "1ahw_abag.yaml"
EMBED_MODEL = "esmc-600m"

# BoltzGen (binder design) — exercised via `tt-bio gen run` on the canonical
# binder fixture (same target the designability accuracy leg + the perf leg use).
# A tiny 1-design job is enough to gate the UX plumbing (progress phases, output
# parses, CLI shape); it is not an accuracy or perf measurement.
GEN_MODEL = "boltzgen"
GEN_SPEC = REPO_ROOT / "examples" / "binder.yaml"
GEN_PROTOCOL = "protein-anything"
GEN_NUM_DESIGNS = 1
GEN_TIMEOUT_S = 1200  # design + refold + analysis for 1 design; load dominates

# esmc embed input: trpcage's 20-mer as a one-sequence FASTA, written into the
# per-run tmp dir so the gate is self-contained (no examples/FASTA dependency).
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


def _run(cmd: list[str], *, env: dict | None = None, timeout: int | None = None,
         cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout+stderr. Raises TimeoutExpired on timeout."""
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


def _check_progress(events: list[dict], model: str) -> list[str]:
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


def _check_npz(npz: Path, seq_id: str, seq: str) -> list[str]:
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

    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "--help"],
                 env=_subprocess_env(), timeout=60)
        if r.returncode != 0:
            problems.append(f"tt-bio --help exited {r.returncode}")
    except Exception as e:
        problems.append(f"tt-bio --help failed to run: {e}")

    # `tt-bio gen run` is a click subcommand that forwards its args to BoltzGen's
    # own argparse parser (tt_bio/boltzgen/cli/boltzgen.py). Click intercepts
    # `--help` at the `gen` wrapper level, so `tt-bio gen run --help` shows the
    # wrapper's short help rather than the run flags — the real flag surface is
    # the forwarded parser's help. Assert the wrapper responds to --help cleanly
    # AND the forwarded parser lists the core design flags a user would reach for.
    try:
        r = _run([sys.executable, "-m", "tt_bio.main", "gen", "run", "--help"],
                 env=_subprocess_env(), timeout=60)
        if r.returncode != 0:
            problems.append(f"gen run --help exited {r.returncode}")
    except Exception as e:
        problems.append(f"gen run --help failed to run: {e}")
    try:
        r = _run([sys.executable, "-m", "tt_bio.boltzgen.cli.boltzgen",
                  "run", "--help"], env=_subprocess_env(), timeout=60)
    except Exception as e:
        problems.append(f"boltzgen run --help failed to run: {e}")
    else:
        if r.returncode != 0:
            problems.append(f"boltzgen run --help exited {r.returncode}")
        else:
            for flag in ("--num_designs", "--protocol", "--output", "--devices",
                         "--budget"):
                if flag not in r.stdout:
                    problems.append(f"boltzgen run --help missing flag {flag}")
    return problems


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
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cap_path = base / f"events_{model}.jsonl"
    cap_path.unlink(missing_ok=True)
    results_path = out_dir / f"boltz_results_{name}" / "results.json"
    struct_dir = out_dir / f"boltz_results_{name}" / "structures"

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
    prog_problems = _check_progress(events, model)
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


def run_embed(model: str, base: Path) -> dict:
    """Run esmc embed on a tiny sequence and gate the UX legs (embed has no fold
    phases — its user-facing progress is the load → embed → done stdout lines)."""
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    seq_id = "tiny"
    fasta = out_dir / "tiny.fasta"
    fasta.write_text(f">{seq_id}\n{EMBED_SEQ}\n")

    cmd = [
        sys.executable, "-m", "tt_bio.main", "embed", str(fasta),
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
    parse_problems = _check_npz(npz, seq_id, EMBED_SEQ) if npz.exists() else [
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
    import re
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
                "(gen --debug --log progress not wired?)"]
    names = [n for _, n in starts]
    # protein-anything runs: design → inverse_folding → folding → design_folding
    # → analysis → filtering (design_folding is the isolated refold = the
    # designability metric's source). Require the headline design + refold +
    # analysis stages so a regression that skips or reorders a phase fails.
    for required in ("design", "analysis"):
        if required not in names:
            problems.append(f"'{required}' stage MISSING from gen progress "
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
    """Run one tiny ``tt-bio gen run`` binder-design job and gate the three UX
    legs (progress phases, output parses, results shape). Returns a result row."""
    out_dir = base / f"out_{model}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not GEN_SPEC.exists():
        sys.exit(f"missing gen fixture {GEN_SPEC}")

    cmd = [
        sys.executable, "-m", "tt_bio.main", "gen", "run", str(GEN_SPEC),
        "--output", str(out_dir),
        "--num_designs", str(GEN_NUM_DESIGNS),
        "--protocol", GEN_PROTOCOL,
        "--devices", "1",
        "--budget", str(GEN_NUM_DESIGNS),
        "--debug", "--log",   # DebugReporter: plain-text stage events on stdout
    ]
    print(f"\n{'='*70}\n[{model}] gen run {GEN_SPEC.name} "
          f"({GEN_PROTOCOL}, {GEN_NUM_DESIGNS} design)\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "progress": False, "parse": False,
           "metrics": False, "gate": False, "error": None, "checks": []}
    t0 = time.monotonic()
    try:
        proc = _run(cmd, env=_subprocess_env(), timeout=GEN_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        row["error"] = f"gen run timed out after {GEN_TIMEOUT_S}s"
        return row
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        row["error"] = (f"gen run exited {proc.returncode}: "
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
        parse_problems = [f"gen run wrote no CIF under {out_dir}"]
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


# ── driver ─────────────────────────────────────────────────────────────────

def _print_fold_row(r: dict) -> None:
    wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
    verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r["error"] else "FAIL"
    print(f"{r['model']:<16}{'progress':>10}{'parse':>7}{'results':>9}"
          f"{wall:>9}  {verdict}")
    print(f"  prog={r['progress']} parse={r['parse']} results={r['results']}")
    for c in r["checks"]:
        print(f"  {c}")


def _print_embed_row(r: dict) -> None:
    wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
    verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r["error"] else "FAIL"
    print(f"{r['model']:<16}{'progress':>10}{'parse':>7}{'manifest':>9}"
          f"{wall:>9}  {verdict}")
    for c in r["checks"]:
        print(f"  {c}")


def _print_gen_row(r: dict) -> None:
    wall = f"{r['seconds']:.0f}s" if r["seconds"] is not None else "-"
    verdict = "PASS" if r["gate"] else f"FAIL ({r['error']})" if r["error"] else "FAIL"
    print(f"{r['model']:<16}{'progress':>10}{'parse':>7}{'metrics':>9}"
          f"{wall:>9}  {verdict}")
    for c in r["checks"]:
        print(f"  {c}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", action="append",
                    choices=FOLD_MODELS + [EMBED_MODEL, GEN_MODEL],
                    help="Gate only this model (repeatable). Default: all five fold "
                         "models + esmc-600m embed + boltzgen gen run.")
    ap.add_argument("--keep", action="store_true",
                    help="Keep the per-run output dirs under the tmp dir for inspection.")
    ap.add_argument("--cli-only", action="store_true",
                    help="Run ONLY the CLI-behaviour leg (predict/embed --help). No card "
                         "needed — usable in GitHub CI. Skips the on-device legs.")
    args = ap.parse_args()

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
    print(f"\n{'#'*78}\nUX GATE — leg 3: CLI behaves (predict / embed / gen run --help)\n{'#'*78}")
    cli_problems = _check_cli()
    all_pass = not cli_problems
    if cli_problems:
        for p in cli_problems:
            print(f"  ✗ {p}")
    else:
        print("  ✓ predict --help, embed --help, gen run --help, tt-bio --help "
              "all OK and list core flags")
    print(f"{'#'*78}")

    if args.cli_only:
        return 0 if all_pass else 1

    models = args.model or (FOLD_MODELS + [EMBED_MODEL, GEN_MODEL])
    fold_models = [m for m in models if m in FOLD_MODELS]
    embed_models = [m for m in models if m == EMBED_MODEL]
    gen_models = [m for m in models if m == GEN_MODEL]

    if not DATA.exists() and fold_models:
        sys.exit(f"missing gate target {DATA}")
    if not GEN_SPEC.exists() and gen_models:
        sys.exit(f"missing gen fixture {GEN_SPEC}")
    if not fold_models and not embed_models and not gen_models:
        return 0 if all_pass else 1

    base = Path(tempfile.mkdtemp(prefix="ux_gate_", dir=str(REPO_ROOT)))
    try:
        rows = []
        for m in fold_models:
            r = run_fold(m, base)
            rows.append(("fold", r))
            all_pass &= r["gate"]
        for m in embed_models:
            r = run_embed(m, base)
            rows.append(("embed", r))
            all_pass &= r["gate"]
        for m in gen_models:
            r = run_gen(m, base)
            rows.append(("gen", r))
            all_pass &= r["gate"]

        print(f"\n{'#'*78}\nUX GATE — summary (fold fixtures: {DATA.name}"
              f"{f' / {ABAG_DATA.name} (opendde-abag)' if ABAG_DATA.exists() else ''}, "
              f"recyc={RECYCLING_STEPS}, steps={SAMPLING_STEPS}, "
              f"samples={DIFFUSION_SAMPLES}, seed={SEED})\n{'#'*78}")
        for kind, r in rows:
            if kind == "fold":
                _print_fold_row(r)
            elif kind == "embed":
                _print_embed_row(r)
            else:
                _print_gen_row(r)
        print(f"{'#'*78}")
        print("GATE PASS — every surface cleared progress + parse + results/manifest "
              "shape, and the CLI behaves" if all_pass
              else "GATE FAIL — a surface missed a UX leg (see above). A UX regression "
                   "blocks a tag, same standing as an accuracy regression.")
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
