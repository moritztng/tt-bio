#!/usr/bin/env python3
"""Put the TT arm and the upstream-GPU reference arm into one layout `score.py` can read.

`perf/b2z2_fusebias/score.py` is the instrument this campaign already uses for accuracy: per
pseudo-domain RMSD, the hinge angle, superposition-free CA-lDDT, CA-lDDT against the experimental
1HCL, and the seed floor. Reusing it rather than writing a second scorer is the point -- a
different instrument would not be comparable with the readings the budget is made of
(memory `verification-instrument-drift-is-shared-code-drift`).

The reference becomes the scorer's BASELINE arm, so what it prints as "the lever" is main's
deviation from upstream, and what it prints as the seed floor is measured on both arms.

    assemble.py --gpu-dir <dir of {size}_gpuref-s*/>  --gpu-runs runs_298.json,runs_512.json \
                --out-cifdir perf/k10_anchor/cif --out-runs perf/k10_anchor/out/runs.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TT_CIF = REPO / "perf" / "b2z2_cond" / "cif"
TT_RUNS = REPO / "perf" / "b2z2_cond" / "out" / "acc_qb2c1.json"
# `default-s0` is the untouched checkout and its digest equals `cond-s0`, so `cond` IS the
# shipped default. Renamed to `ttmain` here because that is what it stands for downstream.
TT_ARMS = {"cond": "ttmain", "base": "ttbase"}   # base = TT_BIO_DEVICE_CONDITIONING=0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu-dir", type=Path, required=True)
    ap.add_argument("--gpu-runs", required=True, help="comma-separated run jsons")
    ap.add_argument("--out-cifdir", type=Path, required=True)
    ap.add_argument("--out-runs", type=Path, required=True)
    a = ap.parse_args()

    shutil.rmtree(a.out_cifdir, ignore_errors=True)
    a.out_cifdir.mkdir(parents=True)

    tt = json.loads(TT_RUNS.read_text())
    runs, env = [], {"tt": tt["env"]}

    for r in tt["runs"]:
        if r["arm"] not in TT_ARMS or r["tag"].endswith("_r1"):
            continue
        size = r["target"].split("_")[-1]
        src = TT_CIF / f"{size}_{r['arm']}-s{r['seed']}"
        cif = next(src.glob("*.cif"))
        tag = f"{TT_ARMS[r['arm']]}-s{r['seed']}"
        dst = a.out_cifdir / f"{size}_{tag}"
        dst.mkdir()
        shutil.copy(cif, dst / cif.name)
        runs.append({**r, "arm": TT_ARMS[r["arm"]], "tag": tag,
                     "sha256_full": hashlib.sha256(cif.read_bytes()).hexdigest()})

    for p in a.gpu_runs.split(","):
        g = json.loads(Path(p).read_text())
        env.setdefault("gpu", g["env"])
        for r in g["runs"]:
            size = r["target"].split("_")[-1]
            src = a.gpu_dir / f"{size}_{r['tag']}"
            cif = next(src.glob("*.cif"))
            dst = a.out_cifdir / f"{size}_{r['tag']}"
            dst.mkdir(exist_ok=True)
            shutil.copy(cif, dst / cif.name)
            # score.py reports plddt verbatim; the TT side records it 0-1 and boltz's CIF
            # B-factor column is 0-100, so normalise rather than print two scales as one column.
            runs.append({**r, "plddt": round(r["plddt"] / 100.0, 6)})

    a.out_runs.parent.mkdir(parents=True, exist_ok=True)
    a.out_runs.write_text(json.dumps({"doc": "TT main vs upstream boltz 2.2.1 GPU reference",
                                      "env": env, "runs": runs}, indent=1))
    print(f"{len(runs)} runs -> {a.out_runs}")
    for r in runs:
        print(" ", r["target"], r["tag"], r.get("mode", "tt"), r["plddt"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
