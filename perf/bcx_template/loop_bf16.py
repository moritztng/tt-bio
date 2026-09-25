#!/usr/bin/env python3
"""The loop gradient with the template stack swapped for tt-bio's host reference at bfloat16.

`grade.py` puts the device swap's d(loss)/d(sequences) at 1.51x JAX's own distance from the
float64 swap. This is the same arm with bf16 arithmetic and no device, so the 1.51x can be read
against what the dtype alone costs at the loop. No device is opened.
"""
import argparse
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import grade as G                                                      # noqa: E402
import jax                                                             # noqa: E402
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402
from splice import evoformer_on_device                                 # noqa: E402


class TemplatePairStackBF16(G.TemplatePairStackF64):
    def _forward(self, act_np, pair_mask_np):
        with torch.no_grad():
            out = self.model.template.run_pair_stack(
                torch.from_numpy(np.asarray(act_np, np.float32).copy()).to(torch.bfloat16),
                torch.from_numpy(np.asarray(pair_mask_np, np.float32).copy()).to(torch.bfloat16))
        self.calls["forward"] += 1
        return out.float().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--baseline", required=True, help="grade.py's run dir (grade.json, grads.npz)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    base = pathlib.Path(args.baseline)
    settings = B.campaign_settings(overrides=[
        f"campaign_seed={args.seed}", "max_trajectories=1", "validation_model=monomer",
        'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]',
        f"project_folder={out}/project"])
    import afgrad as A
    _, ref = A.load_models(A.DEFAULT_PARAMS, template=True)
    stack = TemplatePairStackBF16(ref["bf16"])

    _, states, losses = B.design_state(settings)
    model = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=G.PARAMS,
                                      models=("model_1_ptm",), num_recycle=1,
                                      key=jax.random.PRNGKey(0),
                                      length_bucket_size=campaign_length_bucket(settings),
                                      max_cache_size=2, dropout=False, trunk="jax")
    with evoformer_on_device(None, template=stack):
        _, g, loss = model.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
        jax.effects_barrier()
    if stack.calls["forward"] == 0:
        raise RuntimeError("the bf16 template stack never ran")
    g = np.concatenate([np.asarray(g[k], np.float64).ravel() for k in sorted(g)])
    grads = np.load(base / "grads.npz")
    arms = json.load(open(base / "grade.json"))["arms"]
    jax_vs_f64 = G.cmp(grads["f64"], grads["jax"])
    rep = {"loss": float(loss), "calls": stack.calls, "g_l2": float(np.linalg.norm(g)),
           "bf16_vs_f64": G.cmp(grads["f64"], g), "jax_vs_f64": jax_vs_f64,
           "device_vs_f64": G.cmp(grads["f64"], grads["device"]),
           "loss_shift_bf16_minus_f64": float(loss) - arms["f64"]["loss"],
           "loss_shift_device_minus_f64": arms["device"]["loss"] - arms["f64"]["loss"],
           "loss_shift_jax_minus_f64": arms["jax"]["loss"] - arms["f64"]["loss"],
           "loadavg_end": os.getloadavg()}
    rep["bf16_over_jax"] = rep["bf16_vs_f64"]["rel_l2"] / jax_vs_f64["rel_l2"]
    rep["device_over_bf16"] = rep["device_vs_f64"]["rel_l2"] / rep["bf16_vs_f64"]["rel_l2"]
    np.savez_compressed(out / "grads_bf16.npz", bf16=g)
    (out / "loop_bf16.json").write_text(json.dumps(rep, indent=1))
    print("loop_bf16", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
