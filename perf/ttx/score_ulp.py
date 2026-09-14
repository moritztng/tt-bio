#!/usr/bin/env python3
"""Score the ULP-perturbation arms against the committed upstream fp32 reference.

Lays `ulp_perturb.py`'s folds out in the shape `roof_shared/assemble.py` expects and hands them to
`perf/b2z2_fusebias/score.py` unmodified, so these readings sit next to the stack A/B's on the same
instrument.

    score_ulp.py --runs <ulp0/folds.json> <ulp1/folds.json> --cifroot <out> --outdir <scoredir>
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--arms", required=True, help="score.py arm order, baseline first")
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    merged = {"doc": __doc__, "env": {}, "runs": []}
    cifdir = a.outdir / "cif"
    shutil.rmtree(cifdir, ignore_errors=True); cifdir.mkdir(parents=True)
    for p in a.runs:
        d = json.loads(p.read_text())
        merged["env"][d["runs"][0]["arm"]] = {k: d[k] for k in ("ttnn", "arch", "ulp", "shape")}
        for r in d["runs"]:
            size = r["target"].split("_")[-1]
            src = p.parent / "cif" / f"{size}_{r['tag']}"
            dst = cifdir / f"{size}_{r['tag']}"; dst.mkdir(parents=True)
            cif = next(src.glob("*.cif")); shutil.copy(cif, dst / cif.name)
            merged["runs"].append(r)
    runs_json = a.outdir / "folds.json"
    runs_json.write_text(json.dumps(merged, indent=1))
    subprocess.check_call([sys.executable, str(REPO / "perf/roof_shared/assemble.py"),
                           "--tt-runs", str(runs_json), "--tt-cif", str(cifdir),
                           "--out-cifdir", str(a.outdir / "scorecif"),
                           "--out-runs", str(a.outdir / "score_runs.json")])
    subprocess.check_call([sys.executable, str(REPO / "perf/b2z2_fusebias/score.py"),
                           str(a.outdir / "scorecif"), "--runs", str(a.outdir / "score_runs.json"),
                           "--split", "298", "--arms", a.arms,
                           "--out", str(a.outdir / "score.json")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
