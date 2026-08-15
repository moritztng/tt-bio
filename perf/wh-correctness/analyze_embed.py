#!/usr/bin/env python3
"""The three embed questions that need two jobs to answer, not one.

`check_embed.py` grades one artifact against the sequence it came from. Three of the
embed cells are only meaningful across artifacts, so they are graded here:

  determinism  the same three sequences, submitted twice: `pooled` must be
               bit-identical. Same weights, same input, inference only -- anything else
               is nondeterminism in the served path.
  distinctness three different sequences in one job must give three different vectors.
               Equal vectors mean the service embedded the same thing three times.
  pooling      `pool: cls` must not return what `pool: mean` returned. Identical output
               under two pooling modes means the knob is ignored.

    analyze_embed.py [--artifacts results/artifacts] [--json results/embed_analysis.json]

Exit 0 = every question answered yes, 1 = at least one FAIL, 2 = the artifacts are not
there to answer with (which is not a service verdict, it is a missing measurement).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

MEAN_CELL = "emb_distinct_esmc-600m"
REPEAT_CELL = "emb_determinism_esmc-600m"
CLS_CELL = "emb_pool_cls_esmc-600m"


def load(cell_dir: Path) -> dict[str, np.ndarray]:
    """id -> pooled vector, from the one .npz per sequence the service returns."""
    out = {}
    for f in sorted(cell_dir.rglob("*.npz")):
        with np.load(f, allow_pickle=True) as z:
            if "pooled" in z:
                out[f.stem] = np.asarray(z["pooled"], dtype=np.float64)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    n = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b / n) if n else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=Path(__file__).parent / "results" / "artifacts")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()

    rep: dict = {"fail": [], "warn": [], "checks": {}}
    runs = {name: load(a.artifacts / name) for name in (MEAN_CELL, REPEAT_CELL, CLS_CELL)}
    rep["checks"]["artifacts_per_cell"] = {k: sorted(v) for k, v in runs.items()}
    if not runs[MEAN_CELL]:
        print(f"NO DATA: {a.artifacts / MEAN_CELL} has no .npz -- run --group embed first")
        return 2

    # determinism
    mean_run, repeat = runs[MEAN_CELL], runs[REPEAT_CELL]
    if not repeat:
        rep["warn"].append(f"{REPEAT_CELL} produced no artifact; determinism unmeasured")
    else:
        det = {}
        for sid, v in mean_run.items():
            w = repeat.get(sid)
            if w is None:
                rep["fail"].append(f"determinism: id {sid} is missing from {REPEAT_CELL}")
                continue
            same = bool(v.shape == w.shape and np.array_equal(v, w))
            det[sid] = {"bit_identical": same,
                        "max_abs_diff": float(np.abs(v - w).max()) if v.shape == w.shape else None}
            if not same:
                rep["fail"].append(
                    f"determinism: {sid} differs between two identical submissions "
                    f"(max |d| {det[sid]['max_abs_diff']})")
        rep["checks"]["determinism"] = det

    # distinctness, within the one job
    ids = sorted(mean_run)
    pair = {}
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            c = cosine(mean_run[x], mean_run[y])
            pair[f"{x}|{y}"] = round(c, 6)
            if np.array_equal(mean_run[x], mean_run[y]):
                rep["fail"].append(f"distinctness: {x} and {y} returned the identical vector")
    rep["checks"]["pairwise_cosine"] = pair

    # the pooling knob
    cls = runs[CLS_CELL]
    if not cls:
        rep["warn"].append(f"{CLS_CELL} produced no artifact; the pooling knob is unmeasured")
    else:
        pool = {}
        for sid, v in mean_run.items():
            w = cls.get(sid)
            if w is None:
                rep["fail"].append(f"pooling: id {sid} is missing from {CLS_CELL}")
                continue
            same = bool(v.shape == w.shape and np.array_equal(v, w))
            pool[sid] = {"identical_to_mean": same,
                         "cosine": round(cosine(v, w), 6) if v.shape == w.shape else None}
            if same:
                rep["fail"].append(f"pooling: {sid} under pool=cls is bit-identical to "
                                   f"pool=mean -- the knob is ignored")
        rep["checks"]["pool_cls_vs_mean"] = pool

    rep["verdict"] = "FAIL" if rep["fail"] else ("WARN" if rep["warn"] else "PASS")
    if a.json:
        a.json.write_text(json.dumps(rep, indent=1))
    print(rep["verdict"])
    for f in rep["fail"]:
        print("  FAIL " + f)
    for w in rep["warn"]:
        print("  WARN " + w)
    if pair:
        print("  pairwise cosine: " + ", ".join(f"{k} {v}" for k, v in pair.items()))
    return 1 if rep["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
