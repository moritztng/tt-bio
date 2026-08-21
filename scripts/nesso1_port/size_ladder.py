#!/usr/bin/env python3
"""Nesso-1 size ladder: what it costs and what it costs in accuracy, per sequence length.

Two things only the ladder can answer, and the tyr48 fixture (61 tokens) cannot:

  1. Whether the fp32 trunk default still costs 2.8x once the pocket crop pins N<=256.
     Only the first of the six trunk passes runs at full N; passes 2-6 run cropped. So
     the fp32 penalty should shrink in relative terms as N grows, and the ladder is where
     that stops being a guess.
  2. Whether the device-vs-torch gap grows with N. It is scored against upstream's own
     run-to-run spread R=0.058, never against zero.

Above 196 tokens the crop binds, so every rung from 256 aa up also exercises the crop
path -- the one the pass-2 ``predict_args`` capture bug would have silently broken.

The noise floor comes first: ``--repeats`` device runs on the same card with the same
input, worst pairwise delta, before any cross-arm number is quoted.

Usage (one device context per process, card pinned by the caller):
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_HOLDER=... <env>/bin/python \
      scripts/nesso1_port/size_ladder.py --rungs aa128,aa256 --out perf/nesso1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from tt_bio.nesso1_input import CLI_PREDICT_ARGS, collate, prepare  # noqa: E402

UPSTREAM_SPREAD = 0.058  # upstream's own run-to-run delta, from the GPU reference task
FEAT_SEED = 20260820

SCALARS = (
    "affinity_pred_value",
    "affinity_pred_value1",
    "affinity_pred_value2",
    "affinity_logits_binary",
    "affinity_probability_binary",
    "entropy_pp",
    "entropy_pl",
    "entropy_ll",
    "entropy_crop_pp",
    "entropy_crop_pl",
    "entropy_crop_ll",
)


def scalars_of(pred: dict) -> dict[str, float]:
    return {k: float(pred[k].reshape(-1)[0]) for k in SCALARS if k in pred}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def worst_pairwise(runs: list[dict[str, float]]) -> tuple[float, str]:
    worst, where = 0.0, ""
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            for k in runs[i]:
                d = abs(runs[i][k] - runs[j][k])
                if d > worst:
                    worst, where = d, k
    return worst, where


def featurize(yaml_path: Path, scratch: Path, ccd: Path | None, esm_cache: Path | None) -> dict:
    ds, _, failed = prepare(
        yaml_path, scratch, ccd_pkl=ccd, num_workers=0, esm_cache=esm_cache
    )
    if failed:
        raise SystemExit(f"preprocessing failed for {failed}")
    torch.manual_seed(FEAT_SEED)  # center_random_augmentation draws off the global RNG
    item = ds[0]
    if item.get("exception"):
        raise SystemExit(f"featurizer raised on {yaml_path.name}")
    return collate(item)


def build(weights: str, use_tt: bool, trunk_fp32: bool, affinity_fp32: bool):
    from tt_bio.nesso1 import Nesso1

    m = Nesso1.from_pretrained(
        weights, use_tenstorrent=use_tt, trunk_fp32=trunk_fp32, affinity_fp32=affinity_fp32
    )
    m.use_kernels = False
    m.predict_args.update(CLI_PREDICT_ARGS)
    return m


def timed(model, feats) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model.predict(feats)
    return scalars_of(pred), time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="aa128,aa256,aa512,aa768",
                    help="comma-separated subdirs of --inputs")
    ap.add_argument("--inputs", type=Path, default=REPO / "perf/nesso1/inputs/ladder")
    ap.add_argument("--scratch", type=Path, default=Path("~/scratch/nesso1/ladder").expanduser())
    ap.add_argument("--weights", default="recursionpharma/nesso")
    ap.add_argument("--ccd", type=Path, default=None)
    ap.add_argument("--esm-cache", type=Path, default=None)
    ap.add_argument("--repeats", type=int, default=3, help="device repeats for the floor")
    ap.add_argument("--arms", default="fp32,bf16",
                    help="trunk dtypes to price; the affinity stacks stay fp32")
    ap.add_argument("--torch-max-tokens", type=int, default=400,
                    help="skip the CPU torch arm above this token count; it is minutes, "
                         "not seconds, and the accuracy trend is set by the smaller rungs")
    ap.add_argument("--out", type=Path, default=REPO / "perf/nesso1")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    rungs = [r.strip() for r in args.rungs.split(",") if r.strip()]
    report = {"gate": "nesso1_size_ladder", "R_upstream_spread": UPSTREAM_SPREAD, "rungs": []}
    out_path = args.out / "size_ladder.json"
    args.out.mkdir(parents=True, exist_ok=True)

    for rung in rungs:
        rung_dir = args.inputs / rung
        yamls = sorted(rung_dir.glob("*.yaml"))
        if len(yamls) != 1:
            raise SystemExit(f"expected exactly one yaml in {rung_dir}, found {len(yamls)}")
        yaml_path = yamls[0]
        scratch = args.scratch / rung
        t0 = time.perf_counter()
        feats = featurize(yaml_path, scratch, args.ccd, args.esm_cache)
        feat_s = time.perf_counter() - t0
        n_tokens = int(feats["token_pad_mask"].shape[-1])
        conf = sorted((scratch / "processed/rdkit_conformers").glob("*.pkl"))
        row = {
            "rung": rung,
            "yaml": yaml_path.name,
            "n_tokens": n_tokens,
            "featurize_s": feat_s,
            "conformer_sha256": {p.name: sha256(p) for p in conf},
            "arms": {},
        }
        print(f"\n=== {rung}: {n_tokens} tokens, featurized in {feat_s:.1f}s ===", flush=True)

        ref = None
        if n_tokens <= args.torch_max_tokens:
            ref, ref_s = timed(build(args.weights, False, True, True), feats)
            row["torch_fp32_s"] = ref_s
            row["torch_scalars"] = ref
            print(f"  torch cpu fp32: {ref_s:.1f}s", flush=True)
        else:
            row["torch_fp32_s"] = None
            print("  torch cpu arm skipped (above --torch-max-tokens)", flush=True)

        for arm in arms:
            try:
                model = build(args.weights, True, arm == "fp32", True)
                runs, times = [], []
                for _ in range(args.repeats):
                    vals, secs = timed(model, feats)
                    runs.append(vals)
                    times.append(secs)
            except Exception as exc:  # noqa: BLE001 - an OOM at the top rung is a result
                row["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  device {arm}: FAILED {type(exc).__name__}: {exc}", flush=True)
                continue
            floor, floor_key = worst_pairwise(runs)
            entry = {
                "trunk": arm,
                "affinity": "fp32",
                "wall_s": times,
                "warm_wall_s": min(times[1:]) if len(times) > 1 else times[0],
                "D_device_solo_floor": floor,
                "D_device_solo_floor_key": floor_key,
                "scalars": runs[0],
            }
            if ref is not None:
                worst, key = 0.0, ""
                for k in ref:
                    d = max(abs(r[k] - ref[k]) for r in runs)
                    if d > worst:
                        worst, key = d, k
                entry["X_device_vs_torch"] = worst
                entry["X_device_vs_torch_key"] = key
                entry["X_over_R"] = worst / UPSTREAM_SPREAD
            row["arms"][arm] = entry
            msg = f"  device {arm}: warm {entry['warm_wall_s']:.2f}s floor {floor:.5f}"
            if ref is not None:
                msg += f" X={entry['X_device_vs_torch']:.4f} ({entry['X_over_R']:.2f}xR)"
            print(msg, flush=True)

        report["rungs"].append(row)
        out_path.write_text(json.dumps(report, indent=2) + "\n")  # survive a mid-ladder kill

    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
