#!/usr/bin/env python3
"""Score one model's cross-model A/B and print the verdict the decision turns on.

    xmodel_verdict.py <outdir> [--model M ...]

Two gates, in this order, because one of them is not a judgement call:

  DIGEST   every warm fold of every arm carries the same CIF sha256 and the same plDDT.
           The AdaLN split is bit-exact by construction -- same ops, same dtypes, same
           values, a different issue order -- and this is what turns "by construction"
           into "per consumer". A miss here fails the model outright.
  PERF     |median(B) - median(A)| against the A/A floor |median(A1) - median(A2)|.
           A regression counts only when it exceeds the floor; anything under it is the
           leg's own noise, which `perf-gate-single-shot-legs-recurring-false-alarm` says
           these models have plenty of.

Digests are compared only inside one output directory, which is one host, one card and one
wheel. Nothing here compares against a digest recorded anywhere else.
"""
import argparse, json, statistics, sys
from pathlib import Path

MODELS = ["boltz2", "esmfold2", "protenix-v2", "opendde", "openfold3"]


def load(d: Path, model: str):
    arms = {}
    for label in ("A1", "B1", "A2", "B2"):
        p = d / f"{model}_{label}.json"
        if p.exists():
            arms[label] = json.loads(p.read_text())
    return arms


def digests(rec):
    """(cif sha, plddt) over the warm folds; the cold fold is discarded like its wall."""
    return {(tuple(sorted(f["cif_sha256"].items())), f["plddt"]) for f in rec["folds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir", type=Path)
    ap.add_argument("--model", action="append", default=None)
    a = ap.parse_args()
    bad = 0
    for model in (a.model or MODELS):
        arms = load(a.outdir, model)
        if len(arms) < 4:
            print(f"{model:12s} INCOMPLETE ({sorted(arms)})")
            bad += 1
            continue
        walls = {k: [f["fold_s"] for f in v["folds"]] for k, v in arms.items()}
        mA = statistics.median(walls["A1"] + walls["A2"])
        mB = statistics.median(walls["B1"] + walls["B2"])
        floor = abs(statistics.median(walls["A1"]) - statistics.median(walls["A2"]))
        delta = mB - mA
        seen = set().union(*(digests(v) for v in arms.values()))
        grids = {tuple(v["grid"]) for v in arms.values()}
        wheels = {v["ttnn"] for v in arms.values()}
        ok_d = len(seen) == 1
        ok_p = delta <= floor            # a speedup is always fine
        bad += not (ok_d and ok_p)
        print(f"{model:12s} A {mA:8.3f}  B {mB:8.3f}  delta {delta:+7.3f}  "
              f"A/A floor {floor:6.3f}  digest {'SAME' if ok_d else 'MOVED'}  "
              f"perf {'ok' if ok_p else 'REGRESSED'}")
        if not ok_d:
            for sha, plddt in sorted(seen):
                print(f"             {dict(sha)} plddt={plddt}")
        print(f"             grid {grids} ttnn {wheels} reach "
              f"{ {k: v['reach'] for k, v in arms.items() if any(v['reach'].values())} }")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
