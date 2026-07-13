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
  * esmc-600m embed — seq/s on a fixed batch of 8 ubiquitin-length sequences
    (batch_size 8). Same warmup-then-time protocol.

Baselines live in ``docs/perf_baselines.json`` and are EXPLICIT: an intentional
perf change (landed optimization, deliberate accuracy/perf tradeoff) updates the
baseline via ``--update-baseline --note "<why>"`` — never silently. A regression
the author didn't intend fails the gate. Cover new models as they ship by adding
a spec here + a baseline entry.

Usage::

    # run the whole gate on the card (one device context per model subprocess)
    TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> python3 scripts/perf_regression.py

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
UBIQUITIN = ("MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTL"
             "LHLVLRLRGG")  # 76 aa — tests/test_esmc.py / scripts/esmc_embed_parity.py golden

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
    "esmc-600m":      dict(kind="embed", unit="seq/s", direction="higher",
                           batch_size=8, n_seqs=8),
}
DEFAULT_MODELS = list(SPECS)
FOLD_MODELS = [m for m, s in SPECS.items() if s["kind"] == "fold"]
EMBED_MODELS = [m for m, s in SPECS.items() if s["kind"] == "embed"]

# Light fold protocol — fast, exercises the full trunk + diffusion + heads path.
RECYCLING_STEPS = 1
SAMPLING_STEPS = 10
DIFFUSION_SAMPLES = 1
WARMUP = 2          # warmup folds absorb first-kernel compile (excluded from timing)
REPEAT = 5          # timed folds; report the median
DEFAULT_THRESHOLD = 15.0   # % regression allowed before the gate fails


# ── baseline file ──────────────────────────────────────────────────────────

def load_baselines() -> dict:
    if not BASELINE_FILE.exists():
        return {"models": {}}
    return json.loads(BASELINE_FILE.read_text())


def save_baselines(data: dict) -> None:
    BASELINE_FILE.parent.mkdir(parents=True, exist_ok=True)
    BASELINE_FILE.write_text(json.dumps(data, indent=2) + "\n")


def _version() -> str:
    try:
        import importlib.metadata as md
        return md.version("tt-bio")
    except Exception:
        # not installed (worktree run via PYTHONPATH) — read pyproject
        import re
        txt = (REPO_ROOT / "pyproject.toml").read_text()
        m = re.search(r'^version\s*=\s*"([^"]+)"', txt, re.M)
        return m.group(1) if m else "unknown"


# ── in-process measurement (runs in a child subprocess, one device context) ─

def _boltz_conf_kwargs() -> dict:
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


def measure(model: str, out_path: Path) -> dict:
    """Load one model, warmup, time REPEAT folds, write a JSON result to out_path.

    Runs in its own subprocess (see _run_measure) so each model gets a fresh
    device context — model weights are released cleanly and we avoid the
    cross-model device-reopen path that the worker loop deliberately never takes.
    """
    spec = SPECS[model]
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
    env.setdefault("TT_VISIBLE_DEVICES", "1")
    env.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")
    env.setdefault("LOGURU_LEVEL", "WARNING")
    proc = subprocess.run(cmd, env=env)
    if proc.returncode != 0 or not out.exists():
        print(f"[{model}] measurement FAILED (exit {proc.returncode})", file=sys.stderr)
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


def _print_table(rows: list[dict], baselines: dict, threshold: float) -> bool:
    """Print the per-model comparison table. Returns True iff every row passes."""
    all_pass = True
    bm = baselines.get("models", {})
    title = (f"PERF REGRESSION GATE — {', '.join(r['model'] for r in rows)}  "
             f"| threshold ±{threshold:.0f}%  | warm ({WARMUP} warmup + {REPEAT} timed)")
    print(f"\n{'#' * 78}\n{title}\n{'#' * 78}")
    hdr = (f"{'model':<16}{'metric':<16}{'baseline':>11}{'current':>11}"
           f"{'delta':>10}{'verdict':>10}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        b = bm.get(r["model"])
        unit = r["unit"]
        if b is None:
            cur = f"{r['throughput']:.4g}"
            print(f"{r['model']:<16}{unit:<16}{'(none)':>11}{cur:>11}{'n/a':>10}{'NO BASELINE':>10}")
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
    print(f"  hardware: {rows[0]['hardware']}  |  tt-bio {rows[0]['tt_bio_version']}  "
          f"|  input: {rows[0]['input']}")
    print(f"{'#' * 78}")
    print("GATE PASS — no model regressed beyond ±{:.0f}%".format(threshold) if all_pass
          else "GATE FAIL — a model regressed beyond ±{:.0f}% (see above)".format(threshold))
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
    ok = _print_table(rows, baselines, args.threshold)
    return 0 if ok else 1


def _update_baselines(rows: list[dict], args) -> int:
    if not args.note:
        sys.exit("--update-baseline requires --note \"<why this perf change is intended>\"")
    data = load_baselines()
    data.setdefault("models", {})
    hw = None
    for r in rows:
        if r.get("failed"):
            print(f"[{r['model']}] FAILED — not updating its baseline", file=sys.stderr)
            continue
        hw = r["hardware"]
        data["models"][r["model"]] = dict(
            unit=r["unit"], direction=r["direction"], value=r["throughput"],
            latency_ms=r["latency_ms"], input=r["input"],
            sampling_steps=r["sampling_steps"], diffusion_samples=r["diffusion_samples"],
            recycling_steps=r["recycling_steps"], warmup=r["warmup"], repeat=r["repeat"],
            hardware=r["hardware"], tt_bio_version=r["tt_bio_version"],
            date=r["date"], note=args.note,
        )
    data["hardware"] = hw or data.get("hardware", "blackhole")
    data["threshold_pct"] = args.threshold
    data["date"] = date.today().isoformat()
    save_baselines(data)
    print(f"\nWrote {BASELINE_FILE.relative_to(REPO_ROOT)}  ({len(data['models'])} models)")
    print("Review the diff, then commit it with the change that justifies the new numbers.")
    ok = all(not r.get("failed") for r in rows)
    return 0 if ok else 1


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
