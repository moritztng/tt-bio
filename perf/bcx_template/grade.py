#!/usr/bin/env python3
"""GRADED for the template pair stack swap, against float64, at the stack and at the loop.

CAPTURE. One `sequence_gradients` call on BindCraft 2's JAX trunk, at the settings, seed and
bucket `perf/bcx_seam/run_seam.py` runs the loop at, dropout off. The template `layer_stack`
(`modules.py:247`) is wrapped rather than replaced, and the pair going in, the pair coming out
and the `pair_mask` the block closes over are read off the live program. No cotangent is
captured because none exists: BindCraft 2 builds every template feature from `aatype`, which is
`sequence.argmax(-1)`, so the stack is not on the tangent graph and the swapped round's host map
has its thunks in the forward only.

STACK. The two blocks run three ways on the captured pair:
  device    `splice.TemplatePairStackOnDevice._forward`, the exact call the swap makes
  f64       `tt_bio.af2_reference`'s template pair stack promoted to float64
  host bf16 the same reference at bfloat16, the envelope bf16 costs anyway
and the device output and BindCraft 2's own JAX output are graded against the f64 arm.

LOOP. d(loss)/d(sequences) of the whole program with ONLY the template stack changed, three
ways: BindCraft 2's own JAX, the float64 host stack, and the device. The device's distance from
the float64 arm over JAX's own distance from it is the ratio the campaign's 1.1-1.2x band is
about, and it is the number that decides, because it is the gradient the loop steps on.
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
          str(ROOT / "perf" / "bcx_round"), str(ROOT / "perf" / "bcx_extramsa")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax                                                             # noqa: E402
import numpy as np                                                     # noqa: E402
import torch                                                           # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.af.alphafold.model import layer_stack as LS, modules    # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402
from splice import (TemplatePairStackOnDevice, evoformer_on_device,    # noqa: E402
                    find_template_pair_mask)

PARAMS = "/home/ttuser/bcx_e2e/af2_params"
TEMPLATE_BLOCKS = 2


def f32(x):
    return np.asarray(x, dtype=np.float32)


def cmp(ref, x):
    a, b = np.asarray(ref, np.float64).ravel(), np.asarray(x, np.float64).ravel()
    return {"rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)),
            "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)),
            "max_abs": float(np.abs(a - b).max()), "norm_ref": float(np.linalg.norm(a)),
            "norm_ratio": float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-300))}


class TemplatePairStackF64:
    """The two template blocks on the float64 host reference, shaped like the device class so
    `splice.evoformer_on_device` swaps it in exactly the same way."""

    def __init__(self, model, k_template=TEMPLATE_BLOCKS):
        self.model, self.k_template, self.swapped = model, k_template, 0
        self.calls = {"forward": 0}
        self.dropout_seen: set = set()
        self.shapes: set = set()
        self.channels = None

    def _forward(self, act_np, pair_mask_np):
        act = torch.from_numpy(np.asarray(act_np, np.float64).copy())
        mask = torch.from_numpy(np.asarray(pair_mask_np, np.float64).copy())
        with torch.no_grad():
            out = self.model.template.run_pair_stack(act, mask)
        self.calls["forward"] += 1
        return out.float().numpy()

    def as_jax(self):
        return TemplatePairStackOnDevice.as_jax(self)


def capture(settings, dropout):
    """`act_in`, `act_out` and `pair_mask` of BindCraft 2's own template pair stack."""
    cap = {}
    real = LS.layer_stack

    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)

        def choose(fn):
            if not (getattr(fn, "__name__", None) == "block"
                    and int(num_layers) == TEMPLATE_BLOCKS):
                return made(fn)
            inner = made(fn)
            pair_mask = find_template_pair_mask(fn)
            assert pair_mask is not None, "pair_mask not in the template block's closure"

            def spy(x):
                act, sk = x
                jax.debug.callback(lambda a, m: cap.update(
                    {"act_in": f32(a), "pair_mask": f32(m)}), act, pair_mask)
                out, sk2 = inner((act, sk))
                jax.debug.callback(lambda a: cap.update({"act_out": f32(a)}), out)
                return out, sk2
            return spy
        return choose

    _, states, losses = B.design_state(settings)
    bucket = campaign_length_bucket(settings)
    m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                  models=("model_1_ptm",), num_recycle=1,
                                  key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                  max_cache_size=2, dropout=dropout, trunk="jax")
    modules.layer_stack.layer_stack = factory
    try:
        t0 = time.time()
        _, grads, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
        jax.effects_barrier()
        wall = time.time() - t0
    finally:
        modules.layer_stack.layer_stack = real
    g = np.concatenate([np.asarray(grads[k], np.float64).ravel() for k in sorted(grads)])
    return cap, {"loss": float(loss), "seconds": round(wall, 1), "bucket": bucket,
                 "g_sequences_l2": float(np.linalg.norm(g)), "dropout": dropout}, g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="jax,f64,device")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={out}/project"]
    settings = B.campaign_settings(overrides=overrides)
    rep = {"seed": args.seed, "overrides": overrides, "loadavg_start": os.getloadavg()}

    cap, call, g_jax_capture = capture(settings, dropout=False)
    for k in ("act_in", "act_out", "pair_mask"):
        cap[k] = np.squeeze(cap[k])
    rep["capture"] = {**call, "act_in_shape": list(cap["act_in"].shape),
                      "act_out_shape": list(cap["act_out"].shape),
                      "pair_mask_shape": list(cap["pair_mask"].shape),
                      "pair_mask_zero_fraction": float((cap["pair_mask"] == 0).mean())}
    print("capture", json.dumps(rep["capture"]), flush=True)
    np.savez_compressed(out / "capture.npz", **cap)

    import afgrad as A
    import meter as M
    dm, ref = A.load_models(A.DEFAULT_PARAMS, template=True)
    dev = A.Dev(dm.to_device())
    device_stack = TemplatePairStackOnDevice(dev, k_template=TEMPLATE_BLOCKS)
    f64_stack = TemplatePairStackF64(ref["f64"])
    bf16_stack = TemplatePairStackF64(ref["bf16"])
    rep["template_channels"] = device_stack.channels

    clock = M.Clock(1.0)
    clock.start()
    t0 = time.time()
    d_out = device_stack._forward(cap["act_in"], cap["pair_mask"])
    t1 = time.time()
    rep["device_stack"] = {"forward_s": round(t1 - t0, 3), "aiclk_during": clock.window(t0, t1),
                           "note": "first call on a fresh process, includes program compile"}
    f64_out = f64_stack._forward(cap["act_in"], cap["pair_mask"])
    with torch.no_grad():
        bf16_out = ref["bf16"].template.run_pair_stack(
            torch.from_numpy(cap["act_in"]).to(torch.bfloat16),
            torch.from_numpy(cap["pair_mask"]).to(torch.bfloat16)).float().numpy()
    stack = {"device": cmp(f64_out, d_out), "bc2_jax": cmp(f64_out, cap["act_out"]),
             "host_bf16": cmp(f64_out, bf16_out),
             "device_vs_bc2_jax": cmp(cap["act_out"], d_out)}
    stack["device_over_bc2_jax"] = (stack["device"]["rel_l2"]
                                    / max(stack["bc2_jax"]["rel_l2"], 1e-300))
    rep["stack"] = stack
    print("stack", json.dumps({k: (round(v["rel_l2"], 6), round(v["cos"], 6))
                               if isinstance(v, dict) else round(v, 4)
                               for k, v in stack.items()}), flush=True)

    # ---------------------------------------------------------------- the loop's gradient
    _, states, losses = B.design_state(settings)
    bucket = campaign_length_bucket(settings)
    stacks = {"jax": None, "f64": f64_stack, "device": device_stack}
    grads, rep["arms"] = {}, {}
    for arm in args.arms.split(","):
        # trunk="jax" and its own model: BindCraft 2's Evoformer stays its own, only the
        # template stack moves, and a fresh runner cannot reuse another arm's trace.
        m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                      models=("model_1_ptm",), num_recycle=1,
                                      key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                      max_cache_size=2, dropout=False, trunk="jax")
        before = dict(stacks[arm].calls) if stacks[arm] else None
        t0 = time.time()
        with evoformer_on_device(None, template=stacks[arm]):
            _, g, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                              one_hot_weight=0.0, temperature=1.0,
                                              logit_scale=2.0)
            jax.effects_barrier()
        t1 = time.time()
        grads[arm] = np.concatenate([np.asarray(g[k], np.float64).ravel() for k in sorted(g)])
        if stacks[arm] is not None and stacks[arm].calls["forward"] <= before["forward"]:
            raise RuntimeError(f"{arm}: the swapped template stack never ran")
        rep["arms"][arm] = {"loss": float(loss), "g_l2": float(np.linalg.norm(grads[arm])),
                            "seconds": round(t1 - t0, 1),
                            "calls": dict(stacks[arm].calls) if stacks[arm] else None,
                            "aiclk_during": clock.window(t0, t1) if arm == "device" else None}
        print(arm, json.dumps(rep["arms"][arm], default=str), flush=True)
    clock.stop()

    if {"jax", "f64", "device"} <= set(grads):
        rep["loop"] = {"jax_vs_f64": cmp(grads["f64"], grads["jax"]),
                       "device_vs_f64": cmp(grads["f64"], grads["device"]),
                       "device_vs_jax": cmp(grads["jax"], grads["device"])}
        rep["loop"]["device_over_jax"] = (rep["loop"]["device_vs_f64"]["rel_l2"]
                                          / max(rep["loop"]["jax_vs_f64"]["rel_l2"], 1e-300))
        rep["loop"]["loss_shift_device_minus_jax"] = (rep["arms"]["device"]["loss"]
                                                      - rep["arms"]["jax"]["loss"])
        rep["loop"]["loss_shift_f64_minus_jax"] = (rep["arms"]["f64"]["loss"]
                                                   - rep["arms"]["jax"]["loss"])
        print("loop", json.dumps(rep["loop"], indent=1, default=str), flush=True)
    np.savez_compressed(out / "grads.npz", **grads)
    rep["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "0").split(",")[0]))
    rep["loadavg_end"] = os.getloadavg()
    (out / "grade.json").write_text(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
