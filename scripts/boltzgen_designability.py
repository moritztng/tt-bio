#!/usr/bin/env python3
"""BoltzGen designability check — the standard binder-design QA metric.

Designability (a.k.a. self-consistency RMSD / scRMSD) is the metric
RFdiffusion, BindCraft, and BoltzGen's own paper use to decide whether a
designed binder is any good: take the designed binder's *sequence*, re-fold
it **in isolation** (no target, no template), Kabsch-align the refolded
backbone to the originally-designed backbone, and measure CA-RMSD. A low
scRMSD (BoltzGen's paper bar: <2 A strict, <4 A permissive) means the
sequence actually encodes the shape it was designed for = designable. A high
one flags either a bad design or a device-fidelity problem in the fold.

This check already runs *inside* the ``tt-bio gen`` pipeline — it is not
re-implemented here. The ``design_folding`` step (enabled by default for the
``protein-anything`` / ``protein-small_molecule`` protocols) refolds each
design's sequence alone with the folding checkpoint
(``boltz2_conf_final.ckpt`` — the Boltz-2-derived confidence model BoltzGen
ships), and the ``analysis`` step Kabsch-aligns and writes the result to
``aggregate_metrics_analyze.csv`` as:

    designfolding-bb_rmsd              CA/backbone scRMSD of the isolated refold  <-- the metric
    designfolding-bb_designability_rmsd_2   scRMSD <= 2.0 A  (strict pass)
    designfolding-bb_designability_rmsd_4   scRMSD <= 4.0 A  (permissive pass)

(For protocols that skip design_folding — nanobody/antibody/peptide — only
the whole-complex refold ``bb_rmsd_design`` is available; this script falls
back to it and says so.)

Because the refolder is Boltz-2, whose *own* on-device folding accuracy is
independently ground-truth-gated (``scripts/release_gate.py`` boltz2 leg:
CA-RMSD <= 3 A / TM >= 0.75 on 7ROA), a large scRMSD here isolates cleanly:
if the refolder is accurate (separately gated) yet a design refolds poorly,
the fault is design quality or target hardness, not a refold device bug.

Two modes:

    # run the design pipeline on a target, then score designability
    TT_VISIBLE_DEVICES=1 PYTHONPATH=<worktree> \\
        python scripts/boltzgen_designability.py --num_designs 4

    # score an already-completed gen output dir (no device needed)
    python scripts/boltzgen_designability.py --from-output ./binder

Exit code is 0 unless ``--min-pass-rate`` is given and the fraction of
designs clearing ``--sc-threshold`` falls below it (gate mode).
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# The canonical design example: a de-novo protein binder against chain A of
# 7ROA (examples/binder.yaml, protein-anything protocol). Small and fast — the
# same target README documents for `tt-bio gen run`.
DEFAULT_SPEC = REPO_ROOT / "examples" / "binder.yaml"

# BoltzGen's own designability bars (get_fold_metrics: designability_rmsd_2/_4).
STRICT_A = 2.0
PERMISSIVE_A = 4.0

# Preference order for the scRMSD column. The isolated-refold backbone RMSD is
# the true designability metric (design refolded standalone, no target); the
# whole-complex design-region RMSD is the fallback when design_folding is off.
SC_COLUMNS = [
    ("designfolding-bb_rmsd", "isolated refold (standalone, no target)"),
    ("bb_rmsd_design", "whole-complex refold (design region; target present)"),
]


def _run_gen(spec: Path, out: Path, num_designs: int, protocol: str,
             devices: int, budget: int, reuse: bool,
             diffusion_trace: bool = False) -> None:
    """Drive `tt-bio gen run` for one target. Reuses the shipping pipeline."""
    cmd = [
        sys.executable, "-m", "tt_bio.main", "gen", "run", str(spec),
        "--output", str(out),
        "--num_designs", str(num_designs),
        "--protocol", protocol,
        "--devices", str(devices),
        "--budget", str(budget),
    ]
    if reuse:
        cmd.append("--reuse")
    if diffusion_trace:
        cmd.append("--diffusion_trace")
    print(f"[designability] {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    proc = subprocess.run(cmd, cwd=REPO_ROOT)
    dt = time.monotonic() - t0
    if proc.returncode != 0:
        sys.exit(f"gen run exited {proc.returncode} after {dt:.0f}s")
    print(f"[designability] gen run finished in {dt:.0f}s", flush=True)


def _find_metrics_csv(out: Path) -> Path:
    """Locate the analysis metrics table written by the pipeline."""
    hits = sorted(out.rglob("aggregate_metrics_*.csv"))
    if not hits:
        sys.exit(f"no aggregate_metrics_*.csv under {out} — did analysis run?")
    # Prefer the merged top-level table over per-shard copies (shallowest path).
    return min(hits, key=lambda p: len(p.parts))


def _pick_sc_column(df) -> tuple[str, str]:
    for col, desc in SC_COLUMNS:
        if col in df.columns and df[col].notna().any():
            return col, desc
    sys.exit("metrics table has no designability RMSD column "
             f"(looked for {[c for c, _ in SC_COLUMNS]})")


def score(out: Path, sc_threshold: float) -> dict:
    """Harvest per-design scRMSD from the pipeline's analysis output."""
    import pandas as pd

    df = pd.read_csv(_find_metrics_csv(out))
    col, desc = _pick_sc_column(df)
    sc = df[col].astype(float)
    seqlen = (df["designed_sequence"].str.len()
              if "designed_sequence" in df.columns else [None] * len(df))

    rows = []
    for i in range(len(df)):
        rows.append({
            "id": str(df["id"].iloc[i]) if "id" in df.columns else str(i),
            "len": seqlen[i] if seqlen is not None else None,
            "scrmsd": float(sc.iloc[i]),
        })
    n = len(rows)
    return {
        "column": col,
        "column_desc": desc,
        "n": n,
        "rows": rows,
        "min": float(sc.min()),
        "median": float(sc.median()),
        "max": float(sc.max()),
        "pass_strict": float((sc <= STRICT_A).mean()),
        "pass_permissive": float((sc <= PERMISSIVE_A).mean()),
        "pass_threshold": float((sc <= sc_threshold).mean()),
        "sc_threshold": sc_threshold,
    }


def report(res: dict) -> None:
    print(f"\n{'#'*70}")
    print(f"BoltzGen designability — scRMSD from '{res['column']}'")
    print(f"  ({res['column_desc']})")
    print(f"{'#'*70}")
    print(f"{'design':<28}{'len':>5}{'scRMSD (A)':>13}  designable")
    for r in res["rows"]:
        flag = "PASS" if r["scrmsd"] <= STRICT_A else \
               "(<4A)" if r["scrmsd"] <= PERMISSIVE_A else "FAIL"
        ln = f"{r['len']:>5}" if r["len"] is not None else "    -"
        print(f"{r['id']:<28}{ln}{r['scrmsd']:>13.3f}  {flag}")
    print(f"{'-'*70}")
    print(f"n={res['n']}  scRMSD  min {res['min']:.2f} / "
          f"median {res['median']:.2f} / max {res['max']:.2f} A")
    print(f"designable  <=2A: {res['pass_strict']*100:5.1f}%   "
          f"<=4A: {res['pass_permissive']*100:5.1f}%")
    if res["sc_threshold"] not in (STRICT_A, PERMISSIVE_A):
        print(f"pass @ <={res['sc_threshold']}A: "
              f"{res['pass_threshold']*100:5.1f}%")
    print(f"{'#'*70}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC,
                    help="Design spec YAML (default: examples/binder.yaml).")
    ap.add_argument("--from-output", type=Path, default=None, metavar="DIR",
                    help="Skip design; score an existing gen output dir.")
    ap.add_argument("--output", type=Path, default=None, metavar="DIR",
                    help="Where gen run writes (default: ./boltzgen_designability_<spec>).")
    ap.add_argument("--num_designs", type=int, default=4)
    ap.add_argument("--protocol", default="protein-anything",
                    help="Only protein-anything / protein-small_molecule refold "
                         "the design in isolation (true scRMSD).")
    ap.add_argument("--devices", type=int, default=1,
                    help="Card count; pin the physical card with TT_VISIBLE_DEVICES.")
    ap.add_argument("--budget", type=int, default=8,
                    help="Designs kept after filtering (scoring reads the full set).")
    ap.add_argument("--reuse", action="store_true",
                    help="Resume/keep an existing partial run instead of restarting.")
    ap.add_argument("--diffusion_trace", action="store_true",
                    help="Pass --diffusion_trace to gen run (ttnn trace replay of the "
                    "diffusion DiT; lossless). See docs/boltzgen-trace-replay.md.")
    ap.add_argument("--sc-threshold", type=float, default=STRICT_A,
                    help=f"scRMSD pass bar in A (default {STRICT_A}).")
    ap.add_argument("--min-pass-rate", type=float, default=None, metavar="FRAC",
                    help="Gate mode: exit 1 if the fraction of designs clearing "
                         "--sc-threshold is below FRAC.")
    args = ap.parse_args()

    if args.from_output is not None:
        out = args.from_output
        if not out.exists():
            sys.exit(f"--from-output {out} does not exist")
    else:
        if not args.spec.exists():
            sys.exit(f"missing spec {args.spec}")
        out = args.output or (REPO_ROOT / f"boltzgen_designability_{args.spec.stem}")
        _run_gen(args.spec, out, args.num_designs, args.protocol,
                 args.devices, args.budget, args.reuse,
                 getattr(args, "diffusion_trace", False))

    res = score(out, args.sc_threshold)
    report(res)

    if args.min_pass_rate is not None:
        ok = res["pass_threshold"] >= args.min_pass_rate
        print(f"GATE {'PASS' if ok else 'FAIL'} — "
              f"{res['pass_threshold']*100:.1f}% designable @ <={args.sc_threshold}A "
              f"(need >= {args.min_pass_rate*100:.1f}%)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
