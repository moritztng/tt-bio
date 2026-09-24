#!/usr/bin/env python3
"""Price BindCraft 2's own predictor on CPU, at the PD-L1 design state, per gradient step.

`sequence_gradients` is the call `trajectory.py:131` makes 125 times per trajectory. This
times it as BindCraft 2 itself invokes it, monomer `model_1_ptm` so the arm matches the
device trunk's variant, and reports the per-step median plus the projected trajectory.
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import AlphaFoldDesignModel                         # noqa: E402
from bindcraft.prediction import DifferentiableProteinPredictor        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_1_ptm")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--params", default=os.environ.get(
        "BINDCRAFT_AF2_PARAMS", "/home/ttuser/.boltz/af2"))
    ap.add_argument("--out", default=str(HERE / "refprice.json"))
    args = ap.parse_args()

    settings = B.campaign_settings()
    plan = B.stage_plan(settings)
    design_settings, states, losses = B.design_state(settings)
    shape = B.state_shape(states)
    n = sum(sum(chains.values()) for chains in shape.values())

    model = AlphaFoldDesignModel(
        presets=(args.model,), data_dir=args.params, max_cache_size=2,
        models=(args.model,), num_recycle=settings.get("design_recycles", 1),
        subbatch_size=settings.get("subbatch_size", "auto"),
        length_bucket_size=32)
    conforms = isinstance(model, DifferentiableProteinPredictor)

    times, losses_seen = [], []
    for i in range(args.steps):
        t0 = time.time()
        preds, grads, design_loss = model.sequence_gradients(
            states, losses, model=args.model,
            softmax_weight=1.0, one_hot_weight=0.0, temperature=1.0, logit_scale=2.0)
        dl = float(design_loss)
        dt = time.time() - t0
        times.append(dt)
        losses_seen.append(dl)
        print(f"step {i}: {dt:.2f} s  loss {dl:.6f}"
              f"  grads {{{', '.join(f'{k}:{tuple(v.shape)}' for k, v in grads.items())}}}",
              flush=True)

    warm = times[1:] or times
    med = sorted(warm)[len(warm) // 2]
    blob = {"model": args.model, "n_residues": n, "state_shape": shape,
            "stage_plan": plan, "losses": sorted(losses),
            "protocol_conformant": conforms,
            "step_seconds": times, "first_step_includes_compile": True,
            "warm_median_s": med,
            "projected_gradient_stage_s": med * plan["gradient_steps"],
            "design_loss": losses_seen,
            "grad_shapes": {k: list(v.shape) for k, v in grads.items()},
            "host": os.uname().nodename,
            "loadavg": os.getloadavg()}
    pathlib.Path(args.out).write_text(json.dumps(blob, indent=1))
    print(json.dumps({k: blob[k] for k in (
        "n_residues", "warm_median_s", "projected_gradient_stage_s",
        "protocol_conformant", "loadavg")}, indent=1))


if __name__ == "__main__":
    main()
