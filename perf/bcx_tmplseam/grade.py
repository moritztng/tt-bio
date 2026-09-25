#!/usr/bin/env python3
"""The template pair stack graded against float64, and the loop's own gradient with it swapped.

CAPTURE. One `sequence_gradients` call on BindCraft 2's JAX trunk at the row's settings and
seed, with `modules.TemplatePairStack.__call__` wrapped rather than replaced: the wrapper
records the stack's input `pair_act`, its `pair_mask` and its output, then calls BindCraft 2's
own stack. Done twice, dropout off and on.

GRADED runs the two blocks four ways on the captured input:
  device   `splice.TemplatePairStackOnDevice._primal`, the exact call the swap makes
  f64      `tt_bio.af2_reference`'s template `pair_stack` promoted to float64
  f32/bf16 the same reference at float32 and bfloat16, which is the envelope bf16 costs anyway
  BC2      the captured output of BindCraft 2's own stack, in the program's own bfloat16
and reports each arm's relative distance from f64. Dropout is off for GRADED: with it on the
arms draw the same masks but apply them at different points in the rounding, so the question
the grade asks -- is this the same deterministic function -- is asked with it off, and the
dropout arm is reported separately as a check that the mask plumbing reproduces BC2's draw.

ENVELOPE is the number that decides: `d(loss)/d(sequences)` from the SAME program with only
the template pair stack changed, once for BindCraft 2's JAX, once for the device, once for a
float64 torch reference spliced in the same place. A device arm's distance from JAX means
nothing without the float64 arm beside it (`state/bcx-extramsa.md`).
"""
import argparse
import contextlib
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
from bindcraft.af.alphafold.model import modules                       # noqa: E402
from ttbio_predictor import TTBioAlphaFoldDesignModel                  # noqa: E402
import splice                                                          # noqa: E402

PARAMS = "/home/ttuser/bcx_e2e/af2_params"


def f32(x):
    return np.asarray(x, dtype=np.float32)


def rel(a, b):
    """|a - b| / |b|, both as float64, plus the cosine."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    nb, na = np.linalg.norm(b), np.linalg.norm(a)
    return {"rel": float(np.linalg.norm(a - b) / nb) if nb else None,
            "cos": float(a @ b / (na * nb)) if na and nb else None,
            "absmax_diff": float(np.abs(a - b).max()), "norm": float(na)}


class HostReferenceStack:
    """The same seam as `TemplatePairStackOnDevice`, served by torch at a chosen dtype.

    It is the float64 arm of ENVELOPE. The dropout masks arrive as arguments exactly as they do
    for the device arm, so the two arms differ in the stack's arithmetic and in nothing else.
    """

    CONFIG_ORDER = splice.TemplatePairStackOnDevice.CONFIG_ORDER

    def __init__(self, blocks, dtype, k_blocks=2):
        self.blocks, self.dtype, self.k = blocks, dtype, k_blocks
        self.calls = {"primal": 0}
        self.swapped = 0

    def run(self, act_np, mask_np, keeps_np):
        z = torch.from_numpy(np.asarray(act_np, dtype=np.float64).copy()).to(self.dtype)
        mk = torch.from_numpy(np.asarray(mask_np, dtype=np.float64).copy()).to(self.dtype)
        for keep in keeps_np:
            k = np.asarray(keep, dtype=np.float64)
            if not np.array_equal(k, np.ones_like(k)):
                raise NotImplementedError(
                    "the host reference arm runs with dropout off; a non-identity keep mask "
                    "would have to be threaded into the reference block's residual")
        with torch.no_grad():
            for blk in self.blocks:
                z = blk(z, mk)
        return z.to(torch.float64).numpy()

    def _primal(self, act_np, mask_np, *keeps_np):
        self.calls["primal"] += 1
        return self.run(act_np, mask_np, keeps_np).astype(np.float32)

    def as_jax(self):
        def stack(act, mask_2d, *keeps):
            out = jax.pure_callback(
                self._primal, jax.ShapeDtypeStruct(act.shape, jnp.float32),
                act.astype(jnp.float32), mask_2d.astype(jnp.float32),
                *[k.astype(jnp.float32) for k in keeps],
                vmap_method="sequential")
            return out.astype(act.dtype)
        return stack


def run_round(settings, dropout, swap=None, spy=None):
    """One `sequence_gradients` call; `swap` replaces the template stack, `spy` watches it."""
    _, states, losses = B.design_state(settings)
    bucket = campaign_length_bucket(settings)
    m = TTBioAlphaFoldDesignModel(presets=("model_1_ptm",), data_dir=PARAMS,
                                  models=("model_1_ptm",), num_recycle=1,
                                  key=jax.random.PRNGKey(0), length_bucket_size=bucket,
                                  max_cache_size=2, dropout=dropout, trunk="jax")
    real = modules.TemplatePairStack.__call__
    ctx = (splice.template_pair_stack_on_device(swap) if swap is not None
           else contextlib.nullcontext())
    try:
        with ctx:
            if spy is not None:
                modules.TemplatePairStack.__call__ = spy(modules.TemplatePairStack.__call__)
            t0 = time.time()
            _, grads, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                                  one_hot_weight=0.0, temperature=1.0,
                                                  logit_scale=2.0)
            jax.effects_barrier()
            wall = time.time() - t0
    finally:
        modules.TemplatePairStack.__call__ = real
    g = np.concatenate([f32(grads[k]).ravel() for k in sorted(grads)])
    return {"loss": float(loss), "seconds": round(wall, 1), "bucket": bucket,
            "g_sequences": g, "g_sequences_l2": float(np.linalg.norm(g))}


def capture(settings, dropout):
    """The stack's input, mask and output from a real round, plus that round's loss/gradient."""
    cap = {}

    def spy(real):
        def patched(self, pair_act, pair_mask, use_dropout, safe_key=None):
            out = real(self, pair_act, pair_mask, use_dropout, safe_key)
            jax.debug.callback(
                lambda a, b, c, d: cap.setdefault("stack", {
                    "act_in": f32(a), "mask_2d": f32(b), "act_out": f32(c),
                    "use_dropout": np.asarray(d, dtype=np.float32)}),
                pair_act, pair_mask, out, use_dropout)
            cap["calls"] = cap.get("calls", 0) + 1
            return out
        return patched
    return cap, run_round(settings, dropout, spy=spy)


def identity_masks(act, k_blocks=2):
    """`keep / keep_rate` at rate 0: per_row for ops 0, 1 and 4, per_column for 2 and 3."""
    n, c = act.shape[0], act.shape[2]
    per_row = [0, 1, 4]
    return [np.ones((1, n, c) if i in per_row else (n, 1, c), np.float32)
            for _ in range(k_blocks) for i in range(5)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--skip-device", action="store_true")
    ap.add_argument("--envelope", action="store_true",
                    help="also run the three loop-level programs (device, f64, BC2's JAX)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1", "validation_model=monomer",
                 'design_models=["model_1_ptm"]', 'validation_models=["model_2_ptm"]']
    from bindcraft.settings import parse_setting_overrides, read_settings
    from bindcraft.preflight import cleaned_campaign_settings
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    report = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
              "seed": args.seed, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
              "loadavg": os.getloadavg(), "omp": os.environ.get("OMP_NUM_THREADS")}

    caps = {}
    for tag, drop in (("dropout0", False), ("dropout1", True)):
        cap, got = capture(settings, drop)
        caps[tag] = cap
        report[f"round_{tag}"] = {k: v for k, v in got.items() if k != "g_sequences"}
        report[f"round_{tag}"]["template_stack_calls"] = cap.get("calls")
        st = cap["stack"]
        report[f"round_{tag}"]["stack_shapes"] = {
            "act_in": list(st["act_in"].shape), "mask_2d": list(st["mask_2d"].shape),
            "act_out": list(st["act_out"].shape),
            "use_dropout": float(np.asarray(st["use_dropout"]).ravel()[0])}
    np.savez_compressed(out / "capture.npz",
                        **{f"{t}_{k}": v for t, c in caps.items() for k, v in c["stack"].items()})

    a0, a1 = caps["dropout0"]["stack"]["act_in"], caps["dropout1"]["stack"]["act_in"]
    report["input_is_same_in_both_dropout_arms"] = bool(np.array_equal(a0, a1))
    report["output_differs_across_dropout_arms"] = rel(
        caps["dropout1"]["stack"]["act_out"], caps["dropout0"]["stack"]["act_out"])

    st = caps["dropout0"]["stack"]
    act, mask = st["act_in"], st["mask_2d"]
    ones = identity_masks(act)

    from tt_bio.af2_reference import load_af2_model
    from tt_bio.af2_weights import load_af2_state_dict
    state = load_af2_state_dict(PARAMS)
    arms, refs = {}, {}
    for name, dt in (("f64", torch.float64), ("f32", torch.float32), ("bf16", torch.bfloat16)):
        model = load_af2_model(state, template=True, trunk_dtype=dt)
        model = model.double() if dt == torch.float64 else model
        for p in model.parameters():
            p.requires_grad_(False)
        refs[name] = model
        arms[name] = HostReferenceStack(model.template.pair_stack, dt).run(act, mask, ones)
    arms["bc2_jax"] = st["act_out"].astype(np.float64)

    tmpl = None
    if not args.skip_device:
        import afgrad as _A
        from tt_bio.af2 import load_af2_device_model
        dm = load_af2_device_model(state, template=True, trunk_dtype=torch.bfloat16)
        dev = _A.Dev(dm)
        tmpl = splice.TemplatePairStackOnDevice(dev, k_blocks=2)
        arms["device"] = tmpl._primal(act, mask, *ones).astype(np.float64)
        report["device_dropout_seen"] = dict(tmpl.dropout_seen)

    report["graded_vs_f64"] = {k: rel(v, arms["f64"]) for k, v in arms.items() if k != "f64"}
    report["f64_norm"] = float(np.linalg.norm(arms["f64"]))

    if args.envelope:
        env = {}
        base = run_round(settings, False)
        env["bc2_jax"] = {"loss": base["loss"], "g_l2": base["g_sequences_l2"],
                          "seconds": base["seconds"]}
        f64_arm = run_round(settings, False,
                            swap=HostReferenceStack(refs["f64"].template.pair_stack,
                                                    torch.float64))
        env["f64"] = {"loss": f64_arm["loss"], "g_l2": f64_arm["g_sequences_l2"],
                      "seconds": f64_arm["seconds"],
                      "vs_f64": rel(f64_arm["g_sequences"], f64_arm["g_sequences"])}
        env["bc2_jax"]["vs_f64"] = rel(base["g_sequences"], f64_arm["g_sequences"])
        if tmpl is not None:
            dev_arm = run_round(settings, False, swap=tmpl)
            env["device"] = {"loss": dev_arm["loss"], "g_l2": dev_arm["g_sequences_l2"],
                             "seconds": dev_arm["seconds"],
                             "vs_f64": rel(dev_arm["g_sequences"], f64_arm["g_sequences"]),
                             "calls": dict(tmpl.calls)}
            env["ratio_device_over_jax"] = (
                env["device"]["vs_f64"]["rel"] / env["bc2_jax"]["vs_f64"]["rel"]
                if env["bc2_jax"]["vs_f64"]["rel"] else None)
        report["envelope"] = env

    report["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    (out / "grade.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
