#!/usr/bin/env python3
"""ZEROGRAD and GRADED for the extra-MSA swap, on BindCraft 2's own state.

CAPTURE. One `sequence_gradients` call on BindCraft 2's JAX trunk, at the settings, seed and
bucket `perf/bcx_seam/run_seam.py` runs the loop at. The extra-MSA `layer_stack` is wrapped,
not replaced: identity `custom_vjp` taps sit on the stack's `msa` and `pair` inputs and on its
`pair` output. Each tap's backward carries its forward value as the residual, so every capture
comes from the one recycle pass that has a gradient (the others are stop_gradient'ed and never
reach a backward):

  pair_in, g_pair_in    the pair entering the stack and JAX's own d(loss)/d(pair_in)
  pair_out, g_pair_out  the pair leaving it and the cotangent BindCraft 2 hands it
  msa_in, g_msa_in      the extra-MSA activations and JAX's own d(loss)/d(msa_in)
  extra_msa_mask, mask_2d, read off the stack function's closure as the swap reads them

Left alone, JAX computes no d(loss)/d(msa_in) at all: the activations are a function of
constants, not of the sequence, so they are off the tangent graph. The probe below forces the
cotangent into existence so it can be read as a number.

ZEROGRAD compares |g_msa_in| with |d(loss)/d(sequences)| from the same call, at dropout on
(what every stage but `harden` runs) and off.

GRADED runs the four blocks three ways on the captured (pair_in, msa_in, masks):
  device    `splice.ExtraMsaOnDevice`: `_primal` (untaped), `_taped` and `_backward` fed
            g_pair_out -- the exact calls the swap makes
  f64       `tt_bio.af2_reference` promoted to float64, the MSA track computed for real on the
            captured activations, VJP by torch autograd with the same cotangent
  host bf16 the same reference at bfloat16, the envelope bf16 costs anyway
and grades the device and BindCraft 2's own JAX output against the f64 arm. Dropout is off for
GRADED: the device blocks have none, so the question is the deterministic function.
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
from bindcraft.af.alphafold.model import layer_stack as LS, modules    # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402
from splice import find_extra_msa_masks                                # noqa: E402

PARAMS = "/home/ttuser/bcx_e2e/af2_params"


def f32(x):
    return np.asarray(x, dtype=np.float32)


def capture(settings, dropout):
    cap = {}

    def tap(name):
        @jax.custom_vjp
        def t(x):
            return x

        def fwd(x):
            return x, x

        def bwd(x, g):
            jax.debug.callback(lambda v, c: cap.update({name: f32(v), "g_" + name: f32(c)}), x, g)
            return (g,)
        t.defvjp(fwd, bwd)
        return t

    t_msa, t_pin, t_pout = tap("msa_in"), tap("pair_in"), tap("pair_out")
    real = LS.layer_stack

    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)

        def choose(fn):
            if getattr(fn, "__name__", None) != "extra_msa_stack_fn":
                return made(fn)
            inner = made(fn)
            masks = find_extra_msa_masks(fn)

            def spy(x):
                act, sk = x
                jax.debug.callback(lambda a, b: cap.update(
                    {"extra_msa_mask": f32(a), "mask_2d": f32(b)}), masks["msa"], masks["pair"])
                # `extra_msa_activations` is built from constants alone (`bindcraft/af2.py:134`),
                # so JAX never differentiates it and a tap on it alone never sees a backward.
                # `0 * pair[0, 0, 0]` changes no value and puts it on the tangent graph, so the
                # tap reports the cotangent the stack itself sends back into the MSA input.
                probe = act["msa"] + 0 * act["pair"][0, 0, 0]
                act = {**act, "msa": t_msa(probe), "pair": t_pin(act["pair"])}
                out, sk2 = inner((act, sk))
                return {**out, "pair": t_pout(out["pair"])}, sk2
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
    g_seq = np.concatenate([f32(grads[k]).ravel() for k in sorted(grads)])
    return cap, {"loss": float(loss), "seconds": round(wall, 1), "bucket": bucket,
                 "grad_keys": {k: list(np.shape(grads[k])) for k in sorted(grads)},
                 "g_sequences_l2": float(np.linalg.norm(g_seq)),
                 "g_sequences_absmax": float(np.abs(g_seq).max())}


def zerograd(cap, seq):
    g = cap["g_msa_in"].astype(np.float64)
    return {"extra_msa_mask_shape": list(cap["extra_msa_mask"].shape),
            "extra_msa_mask_nonzero": int((cap["extra_msa_mask"] != 0).sum()),
            "extra_msa_mask_absmax": float(np.abs(cap["extra_msa_mask"]).max()),
            "msa_in_shape": list(cap["msa_in"].shape),
            "g_msa_in_l2": float(np.linalg.norm(g)), "g_msa_in_absmax": float(np.abs(g).max()),
            "g_msa_in_nonzero": int((g != 0).sum()),
            "g_pair_in_l2": float(np.linalg.norm(cap["g_pair_in"].astype(np.float64))),
            "g_pair_out_l2": float(np.linalg.norm(cap["g_pair_out"].astype(np.float64))),
            "g_sequences_l2": seq["g_sequences_l2"],
            "ratio_g_msa_in_to_g_sequences": float(np.linalg.norm(g) / seq["g_sequences_l2"])}


def cmp(ref, x):
    a, b = np.asarray(ref, np.float64).ravel(), np.asarray(x, np.float64).ravel()
    return {"rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)),
            "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)),
            "max_abs": float(np.abs(a - b).max()), "norm_ref": float(np.linalg.norm(a)),
            "norm_ratio": float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-300))}


def reference(model, cap, dtype):
    """Forward and VJP of the four blocks on the host reference, MSA track included."""
    import torch
    msa = torch.from_numpy(cap["msa_in"]).to(dtype)
    pair = torch.from_numpy(cap["pair_in"]).to(dtype).requires_grad_(True)
    emask = torch.from_numpy(cap["extra_msa_mask"]).to(dtype)
    pmask = torch.from_numpy(cap["mask_2d"]).to(dtype)
    msa = msa.reshape(emask.shape + msa.shape[-1:])
    z = pair
    for blk in model.extra_msa:
        msa, z = blk(msa, z, emask, pmask)
    (g,) = torch.autograd.grad(z, pair, torch.from_numpy(cap["g_pair_out"]).to(z.dtype))
    return z.detach().double().numpy(), g.double().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    # run_seam.py's overrides, so the state is the loop's.
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={out_dir}/project"]
    settings = B.campaign_settings(overrides=overrides)
    report = {"seed": args.seed, "overrides": overrides, "loadavg_start": os.getloadavg()}

    caps = {}
    for dropout in (True, False):
        cap, seq = capture(settings, dropout)
        caps[dropout] = cap
        report[f"zerograd_dropout{int(dropout)}"] = {**zerograd(cap, seq), "call": seq}
        print(f"dropout={dropout}", json.dumps(report[f"zerograd_dropout{int(dropout)}"]),
              flush=True)
    cap = caps[False]
    n = cap["pair_in"].shape[0]
    report["n"] = int(n)
    report["mask_2d_zero_fraction"] = float((cap["mask_2d"] == 0).mean())
    np.savez_compressed(out_dir / "capture_dropout0.npz", **cap)

    import torch
    import afgrad as A
    import meter as M
    from splice import ExtraMsaOnDevice
    dm, ref = A.load_models(A.DEFAULT_PARAMS)
    dev = A.Dev(dm.to_device())
    extra = ExtraMsaOnDevice(dev, k_extra=4)

    clock = M.Clock(1.0)
    clock.start()
    t0 = time.time()
    d_primal = extra._primal(cap["pair_in"], cap["extra_msa_mask"], cap["mask_2d"])
    t1 = time.time()
    d_taped, token = extra._taped(cap["pair_in"], cap["extra_msa_mask"], cap["mask_2d"])
    d_grad = extra._backward(token, cap["g_pair_out"])
    t2 = time.time()
    clock.stop()
    report["device"] = {"primal_s": round(t1 - t0, 3), "taped_fwd_bwd_s": round(t2 - t1, 3),
                        "note": "first call on a fresh process, includes program compile",
                        "aiclk_during": clock.window(t0, t2), "calls": dict(extra.calls),
                        "mask_seen": {**extra.mask_seen,
                                      "shapes": sorted(map(list, extra.mask_seen["shapes"]))}}
    print("device", json.dumps(report["device"], default=str), flush=True)

    arms = {}
    for name, dt in (("f64", torch.float64), ("bf16", torch.bfloat16)):
        t0 = time.time()
        arms[name] = reference(ref[name], cap, dt)
        report[f"{name}_s"] = round(time.time() - t0, 1)
        print(name, report[f"{name}_s"], "s", flush=True)
    f64_z, f64_g = arms["f64"]
    grade = {
        "pair_out": {"device_primal": cmp(f64_z, d_primal), "device_taped": cmp(f64_z, d_taped),
                     "bc2_jax": cmp(f64_z, cap["pair_out"]), "host_bf16": cmp(f64_z, arms["bf16"][0]),
                     "device_primal_vs_bc2_jax": cmp(cap["pair_out"], d_primal)},
        "vjp_pair_in": {"device": cmp(f64_g, d_grad), "bc2_jax": cmp(f64_g, cap["g_pair_in"]),
                        "host_bf16": cmp(f64_g, arms["bf16"][1]),
                        "device_vs_bc2_jax": cmp(cap["g_pair_in"], d_grad)},
    }
    for k, arm in (("pair_out", "device_primal"), ("pair_out", "device_taped"),
                   ("vjp_pair_in", "device")):
        grade[k][f"{arm}_over_bc2_jax"] = (grade[k][arm]["rel_l2"]
                                           / max(grade[k]["bc2_jax"]["rel_l2"], 1e-300))
    report["graded"] = grade
    report["stamp"] = A.stamp(int(os.environ.get("TT_VISIBLE_DEVICES", "3").split(",")[0]))
    report["loadavg_end"] = os.getloadavg()
    (out_dir / "grade.json").write_text(json.dumps(report, indent=1, default=str))
    for k, v in grade.items():
        print(k, json.dumps({a: (round(c["rel_l2"], 6), round(c["cos"], 6))
                             if isinstance(c, dict) else round(c, 4) for a, c in v.items()}),
              flush=True)


if __name__ == "__main__":
    main()
