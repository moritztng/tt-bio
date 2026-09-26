#!/usr/bin/env python3
"""What the extra-MSA stack costs a real BindCraft 2 gradient round, in JAX, on the shipped tree.

This is the OFF arm's cost and therefore the ceiling on what the swap can win. `bcx-seam` put the
stack at 21.69 s of 30.95 host seconds, but it measured `perf/bcx_predictor/splice.py`, which
`bcx-backend` promoted into `tt_bio/bindcraft2.py` and deleted, so that number has never been
re-taken on the surface a user runs. No card is needed to take it: the OFF arm IS BindCraft 2's
own JAX.

HOW IT IS TIMED, without a profiler. `jax.debug.callback` runs in computation order on CPU, so
four callbacks bracket the stack:

  forward    a callback immediately before `inner(...)` and one immediately after it
  backward   the two identity `custom_vjp` taps already used for the gradient grade. JAX runs
             backward in reverse, so `pair_out`'s bwd fires BEFORE the stack's backward and
             `pair_in`'s bwd fires AFTER it; the gap between them is the stack's backward.

Each rep is a full `sequence_gradients` on BindCraft 2's own trunk, so the round wall it is
divided into is the real one. `--reps 3` because one round is not a measurement.
"""
import argparse
import json
import os
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(ROOT / "perf" / "bcx_predictor")):
    if p not in sys.path:
        sys.path.insert(0, p)

import jax                                                             # noqa: E402
import bc2_state as B                                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.af.alphafold.model import layer_stack as LS, modules    # noqa: E402
from tt_bio import bindcraft2 as bc2                                   # noqa: E402

PARAMS = os.environ.get("BCX_AF2_PARAMS", "/home/ttuser/bcx_e2e/af2_params")


def timed_round(m, states, losses):
    """One `sequence_gradients`, with the extra-MSA stack bracketed on both passes."""
    marks = {}

    # Every occurrence, not the first. BindCraft 2 recycles, so the stack can run more than one
    # forward per round and only the differentiated pass reaches a backward; a `setdefault` here
    # times one forward and silently calls it the round's cost.
    def mark(name):
        def cb(_):
            marks.setdefault(name, []).append(time.time())
        return cb

    def tap(name):
        @jax.custom_vjp
        def t(x):
            return x

        def fwd(x):
            return x, None

        def bwd(_res, g):
            jax.debug.callback(mark(name), g[..., :1, :1] if g.ndim >= 2 else g)
            return (g,)
        t.defvjp(fwd, bwd)
        return t

    t_pin, t_pout = tap("bwd_after_stack"), tap("bwd_before_stack")
    real = LS.layer_stack

    def factory(num_layers, *a, **kw):
        made = real(num_layers, *a, **kw)

        def choose(fn):
            if getattr(fn, "__name__", None) != "extra_msa_stack_fn":
                return made(fn)
            marks["num_layers"] = int(num_layers)
            inner = made(fn)

            def spy(x):
                act, sk = x
                act = {**act, "pair": t_pin(act["pair"])}
                jax.debug.callback(mark("fwd_start"), act["pair"][0, 0])
                out, sk2 = inner((act, sk))
                jax.debug.callback(mark("fwd_end"), out["pair"][0, 0])
                return {**out, "pair": t_pout(out["pair"])}, sk2
            return spy
        return choose

    modules.layer_stack.layer_stack = factory
    try:
        t0 = time.time()
        _, _grads, loss = m.sequence_gradients(states, losses, softmax_weight=1.0,
                                               one_hot_weight=0.0, temperature=1.0,
                                               logit_scale=2.0)
        jax.effects_barrier()
        wall = time.time() - t0
    finally:
        modules.layer_stack.layer_stack = real

    starts, ends = marks.get("fwd_start", []), marks.get("fwd_end", [])
    b0, b1 = marks.get("bwd_before_stack", []), marks.get("bwd_after_stack", [])
    fwds = [e - st_ for st_, e in zip(starts, ends)]
    bwds = [a - b for b, a in zip(b0, b1)]
    fwd, bwd = sum(fwds), sum(bwds)
    return {"round_s": round(wall, 3), "loss": float(loss),
            "extra_msa_forwards": len(fwds), "extra_msa_backwards": len(bwds),
            "extra_msa_fwd_each_s": [round(x, 3) for x in fwds],
            "extra_msa_bwd_each_s": [round(x, 3) for x in bwds],
            "extra_msa_fwd_s": round(fwd, 3), "extra_msa_bwd_s": round(bwd, 3),
            "extra_msa_total_s": round(fwd + bwd, 3),
            "share_of_round": round((fwd + bwd) / wall, 4),
            "traced_this_rep": marks.get("num_layers") is not None,
            "num_layers": marks.get("num_layers")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={out_dir}/project"]
    settings = B.campaign_settings(overrides=overrides)
    _, states, losses = B.design_state(settings)
    bucket = campaign_length_bucket(settings)

    stamp = {"host": os.uname().nodename, "device_opened": False,
             "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
             "surface": "tt_bio.bindcraft2.design_model_class(), trunk='jax'",
             "arm": "OFF -- BindCraft 2's own extra-MSA stack in JAX; this is the ceiling on "
                    "what the on-card swap can win",
             "seed": args.seed, "reps": args.reps, "bucket": bucket,
             "omp": os.environ.get("OMP_NUM_THREADS"), "nproc": os.cpu_count(),
             "loadavg_start": os.getloadavg(),
             "started_utc": time.strftime("%FT%TZ", time.gmtime())}
    print(json.dumps(stamp), flush=True)

    cls = bc2.design_model_class()
    m = cls(presets=("model_1_ptm",), data_dir=PARAMS, models=("model_1_ptm",), num_recycle=1,
            key=jax.random.PRNGKey(0), length_bucket_size=bucket, max_cache_size=2,
            dropout=False, trunk="jax", pool=None)

    import bindcraft.af2 as bc2_af2
    rows = []
    for i in range(args.reps):
        # The taps are installed by the layer_stack factory, which runs at TRACE time only.
        # BindCraft 2 memoises the lowered gradient program on the instance
        # (`bindcraft/af2.py:333`) and `RunModel.apply` is a `jax.jit` held per runner
        # (`model.py:96`), so without dropping BOTH the second rep reuses rep 1's program, the
        # factory is never asked again and every mark silently reads 0. That is what the first
        # run of this script did. Each rep therefore re-compiles, and `round_s` includes it.
        m.gradient_compile_cache = bc2_af2.CompiledModelCache(
            m.gradient_compile_cache.max_size)
        m.alphafold_runners = {}
        r = timed_round(m, states, losses)
        r["rep"] = i + 1
        r["load1"] = round(os.getloadavg()[0], 1)
        rows.append(r)
        print(json.dumps(r), flush=True)

    untraced = [x["rep"] for x in rows if not x["traced_this_rep"]]
    if untraced:
        raise SystemExit(
            f"reps {untraced} reused an earlier traced program, so their extra-MSA marks never "
            f"fired and their 0.0 is the instrument, not the stack. Refusing to write a share.")
    # Every rep re-traces here, so all of them carry compile; none is a 'steady state' round.
    steady = rows
    summary = {
        "n_steady": len(steady),
        "round_s_median": round(st.median(x["round_s"] for x in steady), 3),
        "round_s_min": min(x["round_s"] for x in steady),
        "round_s_max": max(x["round_s"] for x in steady),
        "extra_msa_total_s_median": round(st.median(x["extra_msa_total_s"] for x in steady), 3),
        "extra_msa_fwd_s_median": round(st.median(x["extra_msa_fwd_s"] for x in steady), 3),
        "extra_msa_bwd_s_median": round(st.median(x["extra_msa_bwd_s"] for x in steady), 3),
        "share_of_round_median": round(st.median(x["share_of_round"] for x in steady), 4),
        "every_rep_includes_compile": True,
        "forwards_per_round": rows[0]["extra_msa_forwards"],
        "backwards_per_round": rows[0]["extra_msa_backwards"],
        "num_layers": rows[0]["num_layers"],
    }
    # The best the swap could do is hand every one of those seconds to the card for free.
    med_round = summary["round_s_median"]
    med_stack = summary["extra_msa_total_s_median"]
    # Deliberately NOT a speedup: this round runs BindCraft 2's Evoformer in JAX too, so its wall
    # is nothing like the 36.37 s device-Evoformer round the swap is actually read against. The
    # stack's own seconds are the transferable number; the share below is against THIS round.
    summary["stack_seconds_is_the_transferable_number"] = med_stack
    summary["share_is_against_a_full_jax_round_not_the_shipped_one"] = round(
        med_stack / med_round, 4)
    stamp["loadavg_end"] = os.getloadavg()
    stamp["finished_utc"] = time.strftime("%FT%TZ", time.gmtime())
    out = {"stamp": stamp, "summary": summary, "reps": rows}
    (out_dir / "hostshare.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
