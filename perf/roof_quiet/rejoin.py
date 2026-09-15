#!/usr/bin/env python3
"""Re-run roof-budget -> roof-residual -> roof-true on one new attrib capture, scripts unchanged.

The three downstream scripts each hardcode a committed filename under a `perf/` tree, so rather
than edit them this builds a shadow `perf/` and points them at it: every data file is a symlink to
the real one, every `*.py` is a copy (a copy, because `split_units.py` derives its capture
directory from `Path(__file__).resolve()`, which walks back out of a symlink into the real tree),
and only the three regenerated files are real. `roof_budget_table.py`, `residual_census.py` and
`true_floor.py` then run verbatim, with their own arguments.

Nothing here computes a number. It arranges inputs and runs the three scripts.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
ROOT = PERF.parent

BUDGET_JSON = "roof_budget_512_qb2c2.json"
ATTRIB_JSON = "attrib2_512_tip_qb2c2.json"
RESIDUAL_JSON = "residual_512_qb2c2.json"


def shadow(dst: Path, src: Path) -> None:
    """Mirror one directory: *.py copied, everything else symlinked, subdirs recursed."""
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.iterdir():
        q = dst / p.name
        if q.exists() or q.is_symlink():
            continue
        if p.is_dir():
            shadow(q, p)
        elif p.suffix == ".py":
            shutil.copyfile(p, q)
        else:
            q.symlink_to(p.resolve())


def build(tree: Path, attrib: Path, captures: Path) -> Path:
    perf = tree / "perf"
    if tree.exists():
        shutil.rmtree(tree)
    for d in ("roof_budget", "roof_residual", "roof_shape", "roof_arb", "b2x_difflayer"):
        shadow(perf / d, PERF / d)
    rb = perf / "roof_budget"
    for name in (ATTRIB_JSON, BUDGET_JSON):
        (rb / name).unlink(missing_ok=True)
    (rb / ATTRIB_JSON).symlink_to(attrib.resolve())
    caps = rb / "captures"
    if caps.is_symlink() or caps.exists():
        shutil.rmtree(caps) if caps.is_dir() and not caps.is_symlink() else caps.unlink()
    caps.symlink_to(captures.resolve())
    (perf / "roof_residual" / RESIDUAL_JSON).unlink(missing_ok=True)
    return perf


def run(cmd: list[str], log: Path) -> None:
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    log.write_text(r.stdout + ("\n--- stderr ---\n" + r.stderr if r.stderr else ""))
    if r.returncode:
        sys.stderr.write(r.stdout[-4000:] + "\n" + r.stderr[-4000:] + "\n")
        raise SystemExit(f"{cmd[1]} failed rc={r.returncode}, see {log}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--attrib", type=Path, required=True, help="the new attrib capture json")
    ap.add_argument("--captures", type=Path, required=True, help="its captures directory")
    ap.add_argument("--baseline", type=Path, default=None,
                    help="json carrying baseline_summary; defaults to --attrib")
    ap.add_argument("--cell-s", type=float, default=None,
                    help="fold of record; default: this session's own plain median, i.e. no rescale")
    ap.add_argument("--time-stat", choices=("median", "percall", "trimmed"), default="trimmed")
    ap.add_argument("--tag", default="quiet")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    out = a.out or HERE / f"out_{a.tag}_{a.time_stat}"
    out.mkdir(parents=True, exist_ok=True)
    base = a.baseline or a.attrib
    cell = a.cell_s
    if cell is None:
        cell = json.loads(base.read_text())["baseline_summary"]["plain_median_s"]
    perf = build(out / "tree", a.attrib, a.captures)
    py = sys.executable

    run([py, PERF / "roof_budget" / "roof_budget_table.py",
         "--run", perf / "roof_budget" / ATTRIB_JSON, "--baseline", base,
         "--captures", perf / "roof_budget" / "captures",
         "--control", perf / "roof_budget" / "instrument_control.json",
         "--stream", perf / "roof_budget" / "stream_roof2.json",
         "--cell-s", cell, "--time-stat", a.time_stat,
         "--out-json", perf / "roof_budget" / BUDGET_JSON,
         "--out-md", out / "ROOF_BUDGET.md"], out / "roof_budget_table.log")

    run([py, PERF / "roof_residual" / "residual_census.py",
         "--budget", perf / "roof_budget" / BUDGET_JSON,
         "--attrib", perf / "roof_budget" / ATTRIB_JSON,
         "--out-json", perf / "roof_residual" / RESIDUAL_JSON,
         "--out-md", out / "ROOF_RESIDUAL.md"], out / "residual_census.log")

    run([py, PERF / "roof_true" / "true_floor.py", "--perf", perf,
         "--out", out / "true_floor.json"], out / "true_floor.log")

    shutil.copyfile(perf / "roof_budget" / BUDGET_JSON, out / BUDGET_JSON)
    shutil.copyfile(perf / "roof_residual" / RESIDUAL_JSON, out / RESIDUAL_JSON)
    print(f"cell {cell:.3f} s, time-stat {a.time_stat} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
