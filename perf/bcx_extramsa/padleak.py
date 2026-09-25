#!/usr/bin/env python3
"""Does the device Evoformer read the masked part of `pair`?

BindCraft 2 pads the token axis to its length bucket and masks the pad out of `mask_2d`. JAX's
Evoformer is insensitive to what sits in the masked rows and columns of `pair` (the float64 VJP
into them is exactly 0.0, grade.json), so overwriting them must leave the loss and the gradient
unchanged. This runs BindCraft 2's own `sequence_gradients` on grade.py's state (seed 100, n=275,
12 masked) with the Evoformer on card and the masked part of the pair INTO the Evoformer
rewritten per arm, before the device sees it:

  keep    untouched, the program bcx-seeds runs
  zero    masked rows and columns set to 0
  x2      masked rows and columns doubled

An arm that moves the loss or the gradient is the device reading masked content.
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(HERE), str(ROOT), str(ROOT / "perf" / "bcx_predictor"),
          str(ROOT / "perf" / "bcx_afgrad"), str(ROOT / "perf" / "bcx_stack"),
          str(ROOT / "perf" / "bcx_round")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax                                                             # noqa: E402
import numpy as np                                                     # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402
from seqgrad import cmp                                                # noqa: E402

PARAMS = "/home/ttuser/bcx_e2e/af2_params"


def rewrite(mode, pair_np, pair_mask_np):
    if mode == "keep":
        return pair_np
    z = np.array(pair_np, copy=True)
    off = np.asarray(pair_mask_np) == 0
    z[off] = 0.0 if mode == "zero" else 2.0 * z[off]
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="keep,zero,x2")
    ap.add_argument("--ref-npz", required=True, help="grads.npz holding the jax arm")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={out}/project"]
    settings = B.campaign_settings(overrides=overrides)
    _, states, losses = B.design_state(settings)
    bucket = campaign_length_bucket(settings)

    import afgrad as A
    import meter as M
    from splice import EvoformerOnDevice, evoformer_on_device
    dm, _ = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device())
    evo = EvoformerOnDevice(dev, k_evo=48)
    primal, taped = evo._primal, evo._taped
    mode = {"now": "keep", "masked_absmax": 0.0, "masked_frac": None}

    def seen(pair_np, pair_mask_np):
        off = np.asarray(pair_mask_np) == 0
        mode["masked_frac"] = float(off.mean())
        if off.any():
            mode["masked_absmax"] = max(mode["masked_absmax"],
                                        float(np.abs(np.asarray(pair_np)[off]).max()))

    def p2(msa, pair, mask, pm):
        seen(pair, pm)
        return primal(msa, rewrite(mode["now"], pair, pm), mask, pm)

    def t2(msa, pair, mask, pm):
        seen(pair, pm)
        return taped(msa, rewrite(mode["now"], pair, pm), mask, pm)

    evo._primal, evo._taped = p2, t2
    clock = M.Clock(1.0)
    clock.start()
    ref = np.load(args.ref_npz)["jax"]
    rep = {"seed": args.seed, "bucket": bucket, "jax_loss": 8.465317726135254,
           "jax_g_l2": float(np.linalg.norm(ref)), "loadavg_start": os.getloadavg(), "arms": {}}
    grads = {}
    for arm in args.arms.split(","):
        mode["now"] = arm
        m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                      models=("model_1_ptm",), num_recycle=1,
                                      key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                      max_cache_size=2, dropout=False, trunk="device")
        t0 = time.time()
        with evoformer_on_device(evo):
            _, g, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
            jax.effects_barrier()
        t1 = time.time()
        grads[arm] = np.concatenate([np.asarray(g[k], np.float64).ravel() for k in sorted(g)])
        rep["arms"][arm] = {"loss": float(loss), "loss_minus_jax": float(loss) - rep["jax_loss"],
                            "vs_jax": cmp(ref, grads[arm]), "seconds": round(t1 - t0, 1),
                            "aiclk_during": clock.window(t0, t1)}
        print(arm, json.dumps(rep["arms"][arm]), flush=True)
    clock.stop()
    rep["masked_frac"], rep["masked_absmax_in"] = mode["masked_frac"], mode["masked_absmax"]
    arms = list(grads)
    rep["between"] = {f"{b}_vs_{a}": cmp(grads[a], grads[b])
                      for i, a in enumerate(arms) for b in arms[i + 1:]}
    rep["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "3").split(",")[0]))
    rep["loadavg_end"] = os.getloadavg()
    (out / "padleak.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["between"], indent=1), flush=True)


if __name__ == "__main__":
    main()
