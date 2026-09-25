#!/usr/bin/env python3
"""The extra-MSA swap graded at the gradient the loop steps on, against a float64 extra-MSA stack.

seqgrad.py compares the device arms with BindCraft 2's own JAX, which is itself an approximation,
so its distances have no envelope. This keeps BindCraft 2's JAX Evoformer and everything else and
swaps ONLY the extra-MSA stack, three ways, on grade.py's state (seed 100, n=275, 12 masked),
dropout off:

  jax        BindCraft 2's own float32 extra-MSA stack (grads reused from seqgrad.py's run)
  f64        `tt_bio.af2_reference`'s four blocks in float64, MSA track computed for real on the
             captured extra-MSA activations, VJP by torch autograd
  device     `splice.ExtraMsaOnDevice`, the swap this row ships

d(loss)/d(sequences) of `jax` and `device` are both graded against `f64`: the device's distance
over JAX's own is the device/JAX ratio at the loop's gradient, with the extra-MSA stack as the
only thing that differs between the three programs.
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
import jax.numpy as jnp                                                # noqa: E402
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402
from seqgrad import cmp                                                # noqa: E402

PARAMS = "/home/ttuser/bcx_e2e/af2_params"


class ExtraMsaF64:
    """The four extra-MSA blocks on the float64 host reference, shaped like `ExtraMsaOnDevice`
    so `splice.evoformer_on_device` swaps it in the same way.

    The stack's MSA input is `extra_msa_activations` of BindCraft 2's constant extra-MSA
    features, so it is the same array on every call at a given length; it is taken from
    grade.py's capture of this state rather than recomputed, and the shape is checked.
    """

    def __init__(self, model, msa_in, k_extra=4):
        self.model, self.k_extra, self.swapped = model, k_extra, 0
        self.msa = torch.from_numpy(msa_in).double()
        self.calls = {"primal": 0, "taped": 0, "backward": 0}
        self._live = {}

    def _run(self, pair_np, emask_np, pmask_np, grad):
        pair = torch.from_numpy(np.asarray(pair_np, np.float64).copy()).requires_grad_(grad)
        emask = torch.from_numpy(np.asarray(emask_np, np.float64).copy())
        pmask = torch.from_numpy(np.asarray(pmask_np, np.float64).copy())
        assert self.msa.shape[1] == pair.shape[0], (self.msa.shape, pair.shape)
        msa = self.msa.reshape(emask.shape + self.msa.shape[-1:])
        z = pair
        with torch.set_grad_enabled(grad):
            for blk in self.model.extra_msa:
                msa, z = blk(msa, z, emask, pmask)
        return pair, z

    def _primal(self, pair_np, emask_np, pmask_np):
        self.calls["primal"] += 1
        return self._run(pair_np, emask_np, pmask_np, False)[1].float().numpy()

    def _taped(self, pair_np, emask_np, pmask_np):
        self._live.clear()
        leaf, z = self._run(pair_np, emask_np, pmask_np, True)
        self._live[0] = (leaf, z)
        self.calls["taped"] += 1
        return z.detach().float().numpy(), np.int32(0)

    def _backward(self, token, g_np):
        leaf, z = self._live.pop(int(token))
        (g,) = torch.autograd.grad(z, leaf, torch.from_numpy(np.asarray(g_np, np.float64).copy()))
        self.calls["backward"] += 1
        return g.float().numpy()

    def as_jax(self):
        from splice import ExtraMsaOnDevice
        return ExtraMsaOnDevice.as_jax(self)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="device,f64")
    ap.add_argument("--ref-npz", required=True, help="seqgrad.py's grads.npz, holding the jax arm")
    ap.add_argument("--capture", required=True, help="grade.py's capture_dropout0.npz")
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
    from splice import ExtraMsaOnDevice, evoformer_on_device
    dm, ref = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device())
    stacks = {"device": ExtraMsaOnDevice(dev, k_extra=4),
              "f64": ExtraMsaF64(ref["f64"], np.load(args.capture)["msa_in"])}
    clock = M.Clock(1.0)
    clock.start()
    grads = {"jax": np.load(args.ref_npz)["jax"]}
    rep = {"seed": args.seed, "bucket": bucket, "jax_loss": 8.465317726135254,
           "loadavg_start": os.getloadavg(), "arms": {}}
    for arm in args.arms.split(","):
        # trunk="jax": the Evoformer stays BindCraft 2's own, only the extra-MSA stack moves.
        m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                      models=("model_1_ptm",), num_recycle=1,
                                      key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                      max_cache_size=2, dropout=False, trunk="jax")
        t0 = time.time()
        with evoformer_on_device(None, extra_msa=stacks[arm]):
            _, g, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
            jax.effects_barrier()
        t1 = time.time()
        grads[arm] = np.concatenate([np.asarray(g[k], np.float64).ravel() for k in sorted(g)])
        if stacks[arm].calls["backward"] < 1:
            raise RuntimeError(f"{arm}: the swapped extra-MSA stack never ran a backward")
        rep["arms"][arm] = {"loss": float(loss), "loss_minus_jax": float(loss) - rep["jax_loss"],
                            "g_l2": float(np.linalg.norm(grads[arm])), "seconds": round(t1 - t0, 1),
                            "calls": dict(stacks[arm].calls),
                            "aiclk_during": clock.window(t0, t1) if arm == "device" else None}
        print(arm, json.dumps(rep["arms"][arm], default=str), flush=True)
    clock.stop()
    rep["graded"] = {"jax_vs_f64": cmp(grads["f64"], grads["jax"]),
                     "device_vs_f64": cmp(grads["f64"], grads["device"]),
                     "device_vs_jax": cmp(grads["jax"], grads["device"])}
    rep["graded"]["device_over_jax"] = (rep["graded"]["device_vs_f64"]["rel_l2"]
                                        / rep["graded"]["jax_vs_f64"]["rel_l2"])
    rep["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "3").split(",")[0]))
    rep["loadavg_end"] = os.getloadavg()
    np.savez_compressed(out / "grads.npz", **grads)
    (out / "envelope.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep["graded"], indent=1), flush=True)


if __name__ == "__main__":
    main()
