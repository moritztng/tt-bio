#!/usr/bin/env python3
"""The card-free half of GRADED for the extra-MSA swap, on the surface we ship.

`bcx-extramsa` graded this against `perf/bcx_predictor/splice.py`. That file is gone and
`229e945b5` re-wrote the lever onto `tt_bio/bindcraft2.py`, so its grades do not transfer. This
re-takes every arm that does not need silicon, so the carded pass adds only the device arm:

ZEROGRAD  BindCraft 2's own `d(loss)/d(extra_msa)` on a REAL round, against `d(loss)/d(sequences)`
          from the same call, with the extra-MSA mask read on that round. `bcx-e2e` shipped a
          seam that silently zeroed 92 % of a gradient at n=256 and the loss still fell, so a
          falling loss is not the check -- this is.

HOST      the four blocks' forward and VJP on `tt_bio.af2_reference` at float64, float32 and
          bfloat16, plus BindCraft 2's OWN JAX pair output and VJP, all graded against the
          float64 arm. That float64 distance for JAX is the denominator every device ratio is
          read against, and it is a property of BindCraft 2 and the mask rather than of the
          port, so banking it now costs the carded pass nothing.

Everything here runs on BindCraft 2's JAX trunk and torch on CPU. No device is opened; run it
with TT_VISIBLE_DEVICES= and it must still pass.
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_predictor"), str(ROOT / "perf" / "bcx_afgrad"),
          str(ROOT / "perf" / "bcx_round")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax                                                             # noqa: E402
import numpy as np                                                     # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.af.alphafold.model import layer_stack as LS, modules    # noqa: E402
from tt_bio import bindcraft2 as bc2                                   # noqa: E402

PARAMS = os.environ.get("BCX_AF2_PARAMS", "/home/ttuser/bcx_e2e/af2_params")


def f32(x):
    return np.asarray(x, dtype=np.float32)


def capture(settings, dropout):
    """One `sequence_gradients` on BindCraft 2's own trunk, with taps on the stack's edges.

    The extra-MSA `layer_stack` is wrapped rather than replaced, so this is BindCraft 2's
    arithmetic throughout and the capture is the input the swap would have been handed.
    """
    cap = {}

    def tap(name):
        @jax.custom_vjp
        def t(x):
            return x

        def fwd(x):
            return x, x

        def bwd(x, g):
            jax.debug.callback(
                lambda v, c: cap.update({name: f32(v), "g_" + name: f32(c)}), x, g)
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
            masks = bc2.find_extra_msa_masks(fn)

            def spy(x):
                act, sk = x
                jax.debug.callback(lambda a, b: cap.update(
                    {"extra_msa_mask": f32(a), "mask_2d": f32(b)}), masks["msa"], masks["pair"])
                # `extra_msa_activations` is built from constants alone (bindcraft/af2.py:134),
                # so JAX never differentiates it and a tap on it alone never sees a backward.
                # `0 * pair[0,0,0]` changes no value and puts it on the tangent graph, so the
                # tap reports the cotangent the stack itself sends back into the MSA input.
                probe = act["msa"] + 0 * act["pair"][0, 0, 0]
                act = {**act, "msa": t_msa(probe), "pair": t_pin(act["pair"])}
                out, sk2 = inner((act, sk))
                return {**out, "pair": t_pout(out["pair"])}, sk2
            return spy
        return choose

    _, states, losses = B.design_state(settings)
    bucket = campaign_length_bucket(settings)
    cls = bc2.design_model_class()
    m = cls(presets=("model_1_ptm",), data_dir=PARAMS, models=("model_1_ptm",), num_recycle=1,
            key=jax.random.PRNGKey(0), length_bucket_size=bucket, max_cache_size=2,
            dropout=dropout, trunk="jax", pool=None)
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
            "ratio_g_msa_in_to_g_sequences":
                float(np.linalg.norm(g) / seq["g_sequences_l2"])}


def cmp(ref, x):
    a, b = np.asarray(ref, np.float64).ravel(), np.asarray(x, np.float64).ravel()
    return {"rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-300)),
            "cos": float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300)),
            "max_abs": float(np.abs(a - b).max()), "norm_ref": float(np.linalg.norm(a)),
            "norm_ratio": float(np.linalg.norm(b) / max(np.linalg.norm(a), 1e-300))}


def host_arm(model, cap, dtype):
    """Forward and VJP of the four blocks on the host reference, MSA track computed for real."""
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
    ap.add_argument("--capture", default=None,
                    help="reuse a capture_dropout0.npz instead of re-running the round; the "
                         "capture is BindCraft 2's own arithmetic and costs ~270 s per dropout "
                         "arm, so the host grading is worth re-running on its own")
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={out_dir}/project"]
    settings = B.campaign_settings(overrides=overrides)
    report = {"seed": args.seed, "host": os.uname().nodename,
              "device_opened": False, "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
              "surface": "tt_bio.bindcraft2 (design_model_class, trunk='jax')",
              "loadavg_start": os.getloadavg(),
              "started_utc": time.strftime("%FT%TZ", time.gmtime())}

    if args.capture:
        cap = dict(np.load(args.capture))
        report["capture_reused"] = args.capture
    else:
        caps = {}
        for dropout in (True, False):
            cap, seq = capture(settings, dropout)
            caps[dropout] = cap
            report[f"zerograd_dropout{int(dropout)}"] = {**zerograd(cap, seq), "call": seq}
            print(f"dropout={dropout} " + json.dumps(
                report[f"zerograd_dropout{int(dropout)}"]), flush=True)
        cap = caps[False]
        np.savez_compressed(out_dir / "capture_dropout0.npz", **cap)
    report["n"] = int(cap["pair_in"].shape[0])
    report["mask_2d_zero_fraction"] = float((cap["mask_2d"] == 0).mean())

    import torch
    import afgrad as A
    # load_models returns (device model or None, {"f64"|"f32"|"bf16": model}). Its docstring
    # says four values; it returns two.
    _dm, ref = A.load_models(A.DEFAULT_PARAMS, device_arm=False)
    assert _dm is None, "device_arm=False still built a device model"

    z64, g64 = host_arm(ref["f64"], cap, torch.float64)
    arms = {"host_f32": (ref["f32"], torch.float32),
            "host_bf16": (ref["bf16"], torch.bfloat16)}
    graded = {}
    for name, (mdl, dt) in arms.items():
        z, g = host_arm(mdl, cap, dt)
        graded[name] = {"pair_out_vs_f64": cmp(z64, z), "vjp_vs_f64": cmp(g64, g)}
    # BindCraft 2's own JAX, captured on the real round, against the same float64 arm. This is
    # the denominator: a device ratio is read against THIS, never against the other device arm.
    graded["bc2_jax"] = {"pair_out_vs_f64": cmp(z64, cap["pair_out"]),
                         "vjp_vs_f64": cmp(g64, cap["g_pair_in"])}
    report["graded_host"] = graded
    report["f64_norms"] = {"pair_out_l2": float(np.linalg.norm(z64)),
                           "vjp_l2": float(np.linalg.norm(g64))}
    report["loadavg_end"] = os.getloadavg()
    report["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    (out_dir / "grade_host.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(graded, indent=1), flush=True)
    print(json.dumps({k: report[k] for k in ("n", "mask_2d_zero_fraction", "f64_norms")},
                     indent=1), flush=True)


if __name__ == "__main__":
    main()
