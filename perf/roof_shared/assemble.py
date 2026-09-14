#!/usr/bin/env python3
"""Lay the TT arms and the committed upstream reference arms out so `score.py` can read them.

`perf/b2z2_fusebias/score.py` is the instrument the K10/ROOF accuracy readings are made with, and
`k10-p1-accuracy-anchor` scored the reference with it unmodified. Reusing it is the point: a second
scorer would not be comparable with the numbers this task has to sit next to
(`verification-instrument-drift-is-shared-code-drift`).

Arms in the assembled layout:

  gpurefshared-s0   upstream boltz 2.2.1, fp32, CPU draws reseeded at the sampler -- the pair for
                    the TT shared arm
  gpuref-s0..s3     upstream, stock CUDA draws. s0 is the pair for the TT plain arm; all four give
                    the reference's own seed floor, which is a known answer (1.66454 A at 512 aa,
                    0.80128 A at 298 aa) and therefore checks this pipeline before it is trusted
  ttshared-s0       this task's fold, TT_BIO_SHARED_DRAW_SEED=0
  ttplain-s0        this task's fold, variable unset

    assemble.py --tt-runs out/folds.json --tt-cif cif --out-cifdir scorecif \
                --out-runs out/score_runs.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REFCIF = HERE / "refcif"
REFRUNS = HERE / "refruns"
REF_TAGS = ["gpurefshared-s0", "gpuref-s0", "gpuref-s1", "gpuref-s2", "gpuref-s3"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tt-runs", type=Path, required=True)
    ap.add_argument("--tt-cif", type=Path, required=True)
    ap.add_argument("--out-cifdir", type=Path, required=True)
    ap.add_argument("--out-runs", type=Path, required=True)
    a = ap.parse_args()

    shutil.rmtree(a.out_cifdir, ignore_errors=True)
    a.out_cifdir.mkdir(parents=True)

    tt = json.loads(a.tt_runs.read_text())
    runs, env = [], {"tt": tt["env"]}

    for r in tt["runs"]:
        size = r["target"].split("_")[-1]
        cif = next((a.tt_cif / f"{size}_{r['tag']}").glob("*.cif"))
        dst = a.out_cifdir / f"{size}_{r['tag']}"
        dst.mkdir(exist_ok=True)
        shutil.copy(cif, dst / cif.name)
        runs.append(r)

    ref = {}
    for p in sorted(REFRUNS.glob("runs_*.json")):
        g = json.loads(p.read_text())
        env.setdefault("gpu", g["env"])
        for r in g["runs"]:
            ref[(r["target"], r["tag"])] = r

    for size in {r["target"].split("_")[-1] for r in tt["runs"]}:
        target = f"cdk2x2_{size}"
        for tag in REF_TAGS:
            src = REFCIF / f"{size}_{tag}"
            cif = next(src.glob("*.cif"))
            dst = a.out_cifdir / f"{size}_{tag}"
            dst.mkdir(exist_ok=True)
            shutil.copy(cif, dst / cif.name)
            r = dict(ref[(target, tag)])
            # score.py prints plddt verbatim and the TT side records it 0-1 while boltz's CIF
            # B-factor column is 0-100; normalise rather than print two scales in one column.
            r["plddt"] = round(r["plddt"] / 100.0, 6)
            r["sha256"] = hashlib.sha256(cif.read_bytes()).hexdigest()[:16]
            r["cif"] = str(dst / cif.name)
            runs.append(r)

    a.out_runs.parent.mkdir(parents=True, exist_ok=True)
    a.out_runs.write_text(json.dumps(
        {"doc": "TT shared/plain draws vs upstream boltz 2.2.1 fp32", "env": env,
         "runs": runs}, indent=1))
    print(f"{len(runs)} runs -> {a.out_runs}")
    for r in runs:
        print(" ", r["target"], f"{r['tag']:16s}", r.get("mode", "?"), r["plddt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
