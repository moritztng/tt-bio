#!/usr/bin/env python3
"""Can the chiral-gradient term actually run on an inference path?

RF3's diffusion atom encoder has use_chiral_features=True and calls
`calc_chiral_grads_flat_impl`, which lives in _vendor/rf3/loss/loss.py -- a
training-loss utility on the inference path. It calls xyz.requires_grad_(True) and
runs under autocast(enabled=False), so it needs grad ENABLED inside an otherwise
torch.no_grad() forward.

Three things to establish before building around it:
  1. does it run at all under no_grad (i.e. does the port need an enable_grad island)?
  2. is the result finite, and does .nan_to_num() actually matter?
  3. is it cheap enough to run every sampler step?
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture", default="/home/ttuser/rf3_ref_work/fi/rna_denc.pt")
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()

    from tt_bio._vendor.rf3.loss.loss import calc_chiral_grads_flat_impl

    cap = torch.load(args.capture, weights_only=False)
    f = cap["in"][0]
    r = cap["in"][1]
    # Captures taken before the harness stopped blanket-.float()ing integer tensors
    # hand back float indices; restore the dtype the model actually uses.
    centers = f["chiral_centers"].long()
    angles = f["chiral_center_dihedral_angles"].float()
    rep = {"n_atoms": list(r.shape), "n_chiral_centers": int(centers.shape[0])}

    def run(x):
        with torch.autocast("cpu", enabled=False):
            return calc_chiral_grads_flat_impl(
                x.detach().clone().float(), centers, angles, False).nan_to_num()

    # 1. under no_grad, the way an inference forward would call it
    try:
        with torch.no_grad():
            g = run(r)
        rep["runs_under_no_grad"] = True
    except Exception as e:
        rep["runs_under_no_grad"] = False
        rep["no_grad_error"] = f"{type(e).__name__}: {e}"[:160]
        with torch.enable_grad():
            g = run(r)
        rep["runs_under_enable_grad"] = True

    rep["out_shape"] = list(g.shape)
    rep["finite"] = bool(torch.isfinite(g).all())
    rep["absmax"] = round(float(g.abs().max()), 6)
    rep["nonzero_rows"] = int((g.abs().sum(-1) > 0).sum())

    # 2. does nan_to_num actually do anything, or is it defensive?
    with torch.enable_grad(), torch.autocast("cpu", enabled=False):
        raw = calc_chiral_grads_flat_impl(
            r.detach().clone().float(), centers, angles, False)
    rep["raw_had_nonfinite"] = bool(not torch.isfinite(raw).all())

    # 3. cost per sampler step
    with torch.enable_grad():
        t0 = time.perf_counter()
        for _ in range(args.steps):
            run(r)
        rep["ms_per_call"] = round((time.perf_counter() - t0) * 1000 / args.steps, 3)
    rep["ms_for_200_steps"] = round(rep["ms_per_call"] * 200, 1)

    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
