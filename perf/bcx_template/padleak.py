#!/usr/bin/env python3
"""Does anything downstream read the template pair stack's padded rows and columns?

`diag.py` puts almost all of the device's distance from float64 in the 12 padded rows and
columns of the [275, 275] pair (0.214 there, 0.0034 on the real region). That only matters if
the loop reads them. This runs BindCraft 2's own JAX program, template stack included, and
multiplies the stack's output in the padded region by `--scale` (0 zeroes it, 2 doubles it)
while the real region passes through untouched. If loss and d(loss)/d(sequences) are
bit-identical to `grade.py`'s unmodified JAX arm, the padded region is unread and the device's
error there cannot reach the loop. No device is opened.
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
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.af.alphafold.model import layer_stack as LS, modules    # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--scale", type=float, required=True)
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

    real, rewrites = LS.layer_stack, []

    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)

        def choose(fn):
            if not (getattr(fn, "__name__", None) == "block"
                    and int(num_layers) == G.TEMPLATE_BLOCKS):
                return made(fn)
            inner = made(fn)
            m = G.find_template_pair_mask(fn)
            assert m is not None, "pair_mask not in the template block's closure"
            rewrites.append(1)

            def rewrite(x):
                act, sk = inner(x)
                keep = (m != 0)[..., None]
                return jax.numpy.where(keep, act, act * args.scale).astype(act.dtype), sk
            return rewrite
        return choose

    _, states, losses = B.design_state(settings)
    model = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=G.PARAMS,
                                      models=("model_1_ptm",), num_recycle=1,
                                      key=jax.random.PRNGKey(0),
                                      length_bucket_size=campaign_length_bucket(settings),
                                      max_cache_size=2, dropout=False, trunk="jax")
    modules.layer_stack.layer_stack = factory
    try:
        _, g, loss = model.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
        jax.effects_barrier()
    finally:
        modules.layer_stack.layer_stack = real
    g = np.concatenate([np.asarray(g[k], np.float64).ravel() for k in sorted(g)])
    g_base = np.load(base / "grads.npz")["jax"]
    loss_base = json.load(open(base / "grade.json"))["arms"]["jax"]["loss"]
    rep = {"scale": args.scale, "rewrites_traced": len(rewrites), "loss": float(loss),
           "loss_base": loss_base, "loss_bit_identical": float(loss) == loss_base,
           "grad_bit_identical": bool(np.array_equal(g, g_base)),
           "grad_max_abs_diff": float(np.abs(g - g_base).max()),
           "loadavg_end": os.getloadavg()}
    (out / "padleak.json").write_text(json.dumps(rep, indent=1))
    print("padleak", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
