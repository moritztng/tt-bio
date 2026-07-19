#!/usr/bin/env python3
"""Standing accuracy release gate — the on-hardware accuracy leg of RELEASING.md.

For every shipped fold architecture (Boltz-2, ESMFold2, ESMFold2-fast,
Protenix-v2, OpenDDE) this
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

BoltzGen is a *design* model, not a fold model — there is no ground truth to fold
against, so it is gated separately from the four above. Its correctness bar is
designability (self-consistency RMSD, scRMSD): refold each design's sequence in
isolation with Boltz-2 and check the shape reproduces. This is the exact
``scripts/boltzgen_designability.py`` method already validated on this hardware
(docs/boltzgen-designability.md, docs/boltzgen-resident-trunk.md's n=8 parity pass)
— reused here, not re-derived. At n=4 (production 500-step sampling) a full
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
see docs/pharma-benchmark.md's ESMC-6b row). The 6b port is the same code path
as 300m/600m (same Block/Embedding modules, same fused RoPE, head_dim 64), so
300m/600m's default-gate parity is a cheap proxy; the opt-in 6b leg is the
standing on-device confirmation. ``run_esmc_parity`` delegates to
``scripts/esmc6b_embed_parity.py`` for 6b (sharded TE safetensors, no sequence
head — the single-.pth / logits path does not apply).

    # gate everything (five fold models + BoltzGen designability + ESMC embed parity) on card 1
    TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> ESM_ROOT=/path/to/esm \
        python scripts/release_gate.py
    # one leg
    python scripts/release_gate.py --model protenix-v2
    python scripts/release_gate.py --model boltzgen
    python scripts/release_gate.py --model esmc-300m

Exit code 0 iff every requested model PASSES its gate; 1 otherwise. Runs on the
device serially (one card context per run); no CPU shortcut for the fold/design.

This is the *accuracy* leg of the release gate. The *UX* leg lives in
``scripts/ux_regression.py`` (live-progress phases, output parsing, CLI shape) —
see RELEASING.md. The two are independent; both must exit 0 before a tag.
"""

import argparse
import importlib.util
import os
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
NAME = DATA.stem  # "prot" -> results land in boltz_results_prot/

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
#   measured best-conf: Boltz-2 1.55 A / TM 0.93 | ESMFold2 2.28 / 0.83 | Protenix-v2 3.87 / 0.71
MODELS = {
    "boltz2":        {"max_rmsd": 3.0, "min_tm": 0.75},
    "esmfold2":      {"max_rmsd": 4.0, "min_tm": 0.65},
    "esmfold2-fast": {"max_rmsd": 4.5, "min_tm": 0.60},
    "protenix-v2":   {"max_rmsd": 6.0, "min_tm": 0.50},
    "opendde":       {"max_rmsd": 6.0, "min_tm": 0.50},
}

# BoltzGen designability leg — see module docstring. Small n and the target the
# README already documents for `tt-bio gen run`; kept fast enough for a release gate
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
# passes the same bar on-device (see docs/pharma-benchmark.md's ESMC-6b row) but
# its ~13 GB load is too slow for the fast gate, so 300m/600m cover it by default.
ESMC_MIN_PCC = 0.99
ESMC_DEFAULT = ["esmc-300m", "esmc-600m"]
ESMC_OPT_IN = ["esmc-6b"]


def _load_structure_harness():
    """Import tests/test_structure.py by path (tests/ is not an installed package)."""
    path = REPO_ROOT / "tests" / "test_structure.py"
    spec = importlib.util.spec_from_file_location("tt_bio_test_structure", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_designability_harness():
    """Import scripts/boltzgen_designability.py by path — reuse its _run_gen/score,
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


def _results_cifs() -> list[Path]:
    d = REPO_ROOT / f"boltz_results_{NAME}" / "structures"
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
    out = REPO_ROOT / f"boltz_results_{NAME}"
    if out.exists():
        shutil.rmtree(out)  # never score a stale run if this predict crashes

    cmd = [
        sys.executable, "-m", "tt_bio.main", "predict", str(DATA),
        "--model", model,
        "--sampling_steps", str(SAMPLING_STEPS),
        "--diffusion_samples", str(DIFFUSION_SAMPLES),
        "--seed", str(SEED),
        "--use_msa_server",
        "--out_dir", str(REPO_ROOT),
    ] + ((["--fast"] if FAST else [])
          + (["--diffusion_trace"] if (DIFFUSION_TRACE and model == "boltz2") else []))
    print(f"\n{'='*70}\n[{model}] folding {DATA.name} "
          f"({SAMPLING_STEPS} steps, {DIFFUSION_SAMPLES} samples)\n{'='*70}", flush=True)

    row = {"model": model, "seconds": None, "rmsd": None, "tm": None,
           "parse": False, "gate": False, "error": None}
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    row["seconds"] = time.monotonic() - t0
    if proc.returncode != 0:
        row["error"] = f"predict exited {proc.returncode}"
        return row

    cifs = _results_cifs()
    try:
        _parse_gate(cifs)
        row["parse"] = True
    except Exception as e:
        row["error"] = f"CIF parse failed: {e}"
        return row

    # Ground-truth RMSD/TM of the confidence-selected structure (harness reads
    # boltz_results_<NAME>/ and examples/ground_truth_structures/ relative to REPO_ROOT).
    try:
        rmsd, tm = harness.evaluate(NAME)  # no thresholds -> returns numbers, never raises
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
        shutil.rmtree(out)  # never score a stale run if this gen crashes

    print(f"\n{'='*70}\n[boltzgen] designing {BOLTZGEN_SPEC.name} "
          f"({BOLTZGEN_NUM_DESIGNS} designs, {BOLTZGEN_PROTOCOL})\n{'='*70}", flush=True)

    row = {"model": "boltzgen", "seconds": None, "scrmsd_median": None,
           "pass_rate": None, "parse": False, "gate": False, "error": None}
    t0 = time.monotonic()
    try:
        bg._run_gen(BOLTZGEN_SPEC, out, BOLTZGEN_NUM_DESIGNS, BOLTZGEN_PROTOCOL,
                    devices=1, budget=BOLTZGEN_NUM_DESIGNS, reuse=False)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",
                    choices=list(MODELS) + ["boltzgen"] + ESMC_DEFAULT + ESMC_OPT_IN,
                    action="append",
                    help="Gate only this model (repeatable). Default: the five fold "
                         "models + boltzgen + ESMC 300m/600m embed parity. esmc-6b is "
                         "opt-in (slow ~13 GB load).")
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

    models = args.model or list(MODELS) + ["boltzgen"] + ESMC_DEFAULT
    fold_models = [m for m in models if m in MODELS]
    want_boltzgen = "boltzgen" in models
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
