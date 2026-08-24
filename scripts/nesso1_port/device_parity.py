#!/usr/bin/env python3
"""Nesso-1 device parity on Tenstorrent: noise floor first, then the comparison.

Nesso-1's output is a scalar, so there is no RMSD to quote and no hash to gate on.
Upstream itself is not run-to-run reproducible (the GPU reference measured 64/64
affinity values differing, max delta 0.058), and Blackhole is not bit-exact
run-to-run above 128 tokens either. So this script measures the floor before it
measures anything else:

  R  the reference spread. Upstream's own run-to-run delta, 0.058, from the GPU
     reference task. A device delta below R is inside the noise the model already has.
  D  device solo-vs-solo. Repeat the SAME device prediction --repeats times on the
     same card with the same input, and take the worst pairwise delta. This is the
     floor: no comparison against another arm can resolve anything finer.
  X  device vs the torch reference, which is bit-exact against upstream. The number
     that matters, read against D and R rather than against zero.

The affinity ensemble mean is the headline scalar; every reported scalar is scored.

Usage (one device context per process, card pinned by the caller):
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=... \
    <tt-bio env>/bin/python scripts/nesso1_port/device_parity.py --weights <snap>
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from model_parity import CLI_PREDICT_ARGS, load_feats  # noqa: E402

# upstream's own run-to-run spread, measured by the GPU reference task
UPSTREAM_SPREAD = 0.058

# Release floors, applied to the shipped arm (bf16 trunk, fp32 affinity stacks) on the
# tyr48 fixture. They live here, next to the measurement, so the gate and a hand run
# report the same verdict.
#
# X_over_R is the worst of the eleven scalars against the bit-exact torch reference,
# divided by upstream's own featurization-draw spread. Measured 3.4308 at 61 tokens, which
# is the WORST rung for bf16 on the measured ladder (3.43 here against 0.88 at 276 and 1.13
# at 532), so a floor set here is one-sided in the safe direction. 5.0 is 1.46x measured,
# tighter than the ~2x the structure models' RMSD floors carry: X_over_R is already
# normalised by the reference's own noise and the device arm is deterministic, so there is
# no seed-to-seed structure noise for the margin to absorb. Only a real numerics change
# moves it.
MAX_X_OVER_R = 5.0
# Worst spread across the repeats on any scalar. Measured exactly 0.0. Gated at 1e-6 rather
# than == 0.0 deliberately: a release floor should never be a literal equality, because a
# card that silently miscomputes reads as a code regression. bf16's quantum at these
# magnitudes is ~0.008, four orders above this, so real nondeterminism still fails.
MAX_DEVICE_SPREAD = 1e-6

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


def run(model, feats) -> tuple[dict[str, float], float]:
    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model.predict(feats)
    return scalars_of(pred), time.perf_counter() - t0


def worst_delta(runs: list[dict[str, float]]) -> tuple[float, str]:
    worst, where = 0.0, ""
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            for k in runs[i]:
                d = abs(runs[i][k] - runs[j][k])
                if d > worst:
                    worst, where = d, k
    return worst, where


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", type=Path,
                    default=REPO / "scripts/nesso1_port/parity_artifacts/tyr48")
    ap.add_argument("--weights", default="recursionpharma/nesso")
    ap.add_argument("--repeats", type=int, default=3,
                    help="device repeats for the solo-vs-solo floor")
    ap.add_argument("--trunk", choices=("bf16", "fp32"), default="fp32",
                    help="dtype of the 48-block trunk on device; fp32 is the default "
                         "because bf16 costs 5.8x on the affinity value (0.116 vs "
                         "0.020 at 61 tokens) to save 2.8x wall time")
    ap.add_argument("--affinity", choices=("bf16", "fp32"), default="fp32",
                    help="dtype of the two 8-block affinity stacks on device; "
                         "upstream runs them under autocast disabled, i.e. fp32")
    ap.add_argument("--max-x-over-r", type=float, default=MAX_X_OVER_R,
                    help="release floor on the worst scalar vs torch, in units of "
                         "upstream's own run-to-run spread")
    ap.add_argument("--max-device-spread", type=float, default=MAX_DEVICE_SPREAD,
                    help="release floor on device run-to-run spread across --repeats")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from tt_bio.nesso1 import Nesso1

    feats, meta = load_feats(args.fixture.resolve())
    n_tokens = int(feats["token_pad_mask"].shape[-1])

    def build(use_tt: bool):
        m = Nesso1.from_pretrained(
            args.weights,
            use_tenstorrent=use_tt,
            trunk_fp32=args.trunk == "fp32",
            affinity_fp32=args.affinity == "fp32",
        )
        m.use_kernels = False
        m.predict_args.update(CLI_PREDICT_ARGS)
        m.predict_args["recycling_steps"] = meta.get("recycling_steps", 5)
        return m

    # the bit-exact torch arm
    ref, ref_s = run(build(False), feats)

    # the device arm, repeated on the same card, same input
    dev_model = build(True)
    dev_runs, dev_times = [], []
    for _ in range(args.repeats):
        vals, secs = run(dev_model, feats)
        dev_runs.append(vals)
        dev_times.append(secs)

    floor, floor_key = worst_delta(dev_runs)
    rows = []
    worst_vs_ref, worst_key = 0.0, ""
    for k in ref:
        got = [r[k] for r in dev_runs]
        d = max(abs(g - ref[k]) for g in got)
        if d > worst_vs_ref:
            worst_vs_ref, worst_key = d, k
        rows.append({
            "key": k, "torch": ref[k], "device": got,
            "max_abs_delta_vs_torch": d,
            "device_spread": max(got) - min(got),
        })

    report = {
        "gate": "nesso1_device_parity",
        "fixture": args.fixture.name,
        "n_tokens": n_tokens,
        "repeats": args.repeats,
        "arm": {"trunk": args.trunk, "affinity": args.affinity},
        "R_upstream_spread": UPSTREAM_SPREAD,
        "D_device_solo_floor": floor,
        "D_device_solo_floor_key": floor_key,
        "X_device_vs_torch": worst_vs_ref,
        "X_device_vs_torch_key": worst_key,
        "X_over_R": worst_vs_ref / UPSTREAM_SPREAD,
        "wall_s": {"torch": ref_s, "device": dev_times},
        "scalars": rows,
    }
    # Two floors, not one. Bit-exactness is not available on either side, so the
    # accuracy floor is a multiple of the reference's own spread; the determinism floor
    # is separate because a card that starts wandering is a different failure from a
    # numerics change and should not be able to hide inside the accuracy margin.
    spread = max(r["device_spread"] for r in rows) if rows else 0.0
    report["max_device_spread"] = spread
    report["floors"] = {"max_x_over_r": args.max_x_over_r,
                        "max_device_spread": args.max_device_spread}
    report["verdict"] = ("PASS" if report["X_over_R"] <= args.max_x_over_r
                         and spread <= args.max_device_spread else "FAIL")
    text = json.dumps(report, indent=2)
    print(text)
    if args.json:
        args.json.write_text(text + "\n")
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    from tt_bio.device_lease import CONTENDED_EXIT_CODE, DeviceInUseError

    try:
        raise SystemExit(main())
    except DeviceInUseError as exc:
        # A co-tenant on the card is not a parity result. Exit on the reserved code so the
        # release gate reports contention instead of scoring an arm that never ran: this leg
        # failed both v0.7.0 gate passes this way and read as "missed the reference floor or
        # drifted run to run" both times.
        print(f"device contention, no measurement taken: {exc}", file=sys.stderr)
        raise SystemExit(CONTENDED_EXIT_CODE) from None
