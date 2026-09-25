#!/usr/bin/env python3
"""What the extra-MSA swap does to the gradient the loop actually steps on.

One BindCraft 2 state (grade.py's: seed 100, n=275, 12 masked), dropout off, one
`sequence_gradients` call per arm, each on a fresh model so each traces its own program:

  jax        BindCraft 2's own trunk, the reference the campaign grades against
  evo        the Evoformer on card, extra-MSA in JAX -- the program bcx-seeds runs
  evo+extra  both stacks on card

d(loss)/d(sequences) of `evo` and `evo+extra` are graded against `jax`. The question is whether
`evo+extra` sits where `evo` already sits, i.e. whether the swap adds error on top of the swap the
campaign already accepted.
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

import contextlib                                                      # noqa: E402
import jax                                                             # noqa: E402
import numpy as np                                                     # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402

PARAMS = "/home/ttuser/bcx_e2e/af2_params"


def cmp(ref, x):
    a, b = np.asarray(ref, np.float64), np.asarray(x, np.float64)
    return {"rel_l2": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
            "cos": float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))),
            "norm_ratio": float(np.linalg.norm(b) / np.linalg.norm(a))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
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
    from splice import EvoformerOnDevice, ExtraMsaOnDevice, evoformer_on_device
    dm, _ = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device())
    evo = EvoformerOnDevice(dev, k_evo=48)
    extra = ExtraMsaOnDevice(dev, k_extra=4)
    clock = M.Clock(1.0)
    clock.start()

    rep = {"seed": args.seed, "bucket": bucket, "loadavg_start": os.getloadavg(), "arms": {}}
    grads = {}
    for arm in ("jax", "evo", "evo+extra"):
        ctx = (contextlib.nullcontext() if arm == "jax" else
               evoformer_on_device(evo, extra_msa=extra if arm == "evo+extra" else None))
        m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                      models=("model_1_ptm",), num_recycle=1,
                                      key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                      max_cache_size=2, dropout=False,
                                      trunk="jax" if arm == "jax" else "device")
        t0 = time.time()
        with ctx:
            _, g, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
            jax.effects_barrier()
        t1 = time.time()
        grads[arm] = np.concatenate([np.asarray(g[k], np.float64).ravel() for k in sorted(g)])
        rep["arms"][arm] = {"loss": float(loss), "seconds_incl_compile": round(t1 - t0, 1),
                            "g_l2": float(np.linalg.norm(grads[arm])),
                            "aiclk_during": clock.window(t0, t1) if arm != "jax" else None}
        print(arm, json.dumps(rep["arms"][arm], default=str), flush=True)
    clock.stop()
    rep["extra_calls"] = dict(extra.calls)
    rep["extra_mask_seen"] = {**extra.mask_seen,
                              "shapes": sorted(map(list, extra.mask_seen["shapes"]))}
    rep["graded"] = {"evo_vs_jax": cmp(grads["jax"], grads["evo"]),
                     "evo+extra_vs_jax": cmp(grads["jax"], grads["evo+extra"]),
                     "evo+extra_vs_evo": cmp(grads["evo"], grads["evo+extra"])}
    rep["graded"]["ratio_evo+extra_over_evo"] = (rep["graded"]["evo+extra_vs_jax"]["rel_l2"]
                                                 / rep["graded"]["evo_vs_jax"]["rel_l2"])
    rep["loss_delta"] = {"evo": rep["arms"]["evo"]["loss"] - rep["arms"]["jax"]["loss"],
                         "evo+extra": rep["arms"]["evo+extra"]["loss"] - rep["arms"]["jax"]["loss"]}
    rep["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "3").split(",")[0]))
    rep["loadavg_end"] = os.getloadavg()
    np.savez_compressed(out / "grads.npz", **{k.replace("+", "_"): v for k, v in grads.items()})
    (out / "seqgrad.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["graded"], indent=1), json.dumps(rep["loss_delta"]), flush=True)


if __name__ == "__main__":
    main()
