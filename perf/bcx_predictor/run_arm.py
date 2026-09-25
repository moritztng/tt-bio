#!/usr/bin/env python3
"""Run BindCraft 2's own campaign, unmodified, against a chosen predictor.

`campaign.py` takes its predictor as a parameter everywhere and constructs
`AlphaFoldDesignModel` in exactly one place, so an arm is chosen by which class is built,
not by editing the loop. Nothing here changes BindCraft 2's step budget, its stage plan,
its recycles, its filters or its ranking: the campaign runs its own 125 gradient steps a
trajectory and accepts or rejects on its own thresholds.

Two deviations from BindCraft 2's defaults, both deliberate and both named in
state/bcx-predictor.md:

  * the model pool is pinned to the MONOMER checkpoints. tt-bio's AF2 trunk on main is
    monomer `model_1_ptm`; BindCraft 2 samples design models from the multimer_v3 pool.
    Pinning both arms to monomer is what makes the comparison like-for-like, and the
    variant gap belongs to `bcx-multimer`.
  * `max_trajectories` bounds the run. `examples/pdl1.json` sets `number_of_final_designs`
    with no attempt cap, so an unbounded campaign runs until it has ten accepted designs.
    A trajectory BUDGET is not a change to the model's own work; every trajectory that runs
    runs in full.
"""
import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bc2_state as B                                                  # noqa: E402
_ROOT = HERE.parents[1]
for _p in (str(_ROOT), str(_ROOT / "perf" / "bcx_afgrad"), str(_ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import bindcraft.campaign as campaign                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402

MONOMER = ("model_1_ptm", "model_2_ptm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["reference", "control", "device"], default="reference",
                    help="reference: BindCraft 2's own AlphaFoldDesignModel, untouched. "
                         "control: TTBioAlphaFoldDesignModel with trunk='jax' -- our class, "
                         "BindCraft 2's trunk, so a device result is compared against the same "
                         "call path. device: trunk on card.")
    ap.add_argument("--trajectories", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", default=None)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--binder-length", type=int, default=0,
                    help="pin the drawn binder length, so a size can be held while something "
                         "else varies; 0 leaves BindCraft 2 to draw from its own range")
    ap.add_argument("--no-levers", action="store_true",
                    help="skip perf/bcx_stack's lever arming. The levers are PERF levers and "
                         "the pool arm asks a correctness question; they also need "
                         "autograd._via2d, which is bcx-mm2d's and never landed on main, so "
                         "on a main-based branch Levers() cannot construct at all.")
    ap.add_argument("--multimer-pool", action="store_true",
                    help="run the SHIPPED configuration: examples/pdl1.json's five "
                         "model_1..5_multimer_v3 design models, sampled one per gradient step, "
                         "with the device trunk following BindCraft 2's choice. OFF by default "
                         "and off is byte-identical to the monomer-pinned arm every result in "
                         "this campaign was measured on -- bcx-seeds has matched device-vs-JAX "
                         "pairs in flight and a changed default would void the set.")
    ap.add_argument("--pool-resident", type=int, default=0,
                    help="how many of the five trunks stay on card; 0 means all five")
    ap.add_argument("--bucket", type=int, default=1,
                    help="length_bucket_size override; 0 leaves BindCraft 2's own "
                         "default of 32. It used to have to be 1 because the trunk "
                         "refused a masked fold; all three mask sites are in af2.py "
                         "now, so 32 runs and is the lab's configuration.")
    args = ap.parse_args()

    project = args.out or str(HERE / "runs" / f"{args.arm}_seed{args.seed}")
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", f"max_trajectories={args.trajectories}",
                 *([f"binder_lengths=[{args.binder_length}]"] if args.binder_length else []),
                 f"project_folder={project}"]
    if not args.multimer_pool:
        # The monomer pin, unchanged. Every number this campaign has published was measured
        # with these three lines in place, so they stay the default.
        overrides += ["validation_model=monomer", 'design_models=["model_1_ptm"]',
                      'validation_models=["model_2_ptm"]']
    if args.bucket:
        overrides.append(f"length_bucket_size={args.bucket}")
    settings = cleaned_campaign_settings(
        read_settings(args.settings or os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    if not args.multimer_pool:
        # Both arms on the monomer checkpoints -- see the module docstring.
        campaign.MULTIMER_POOL = MONOMER

    pool = None
    if args.multimer_pool and args.arm == "device":
        import multimer_pool as MP
        pool = MP.MultimerPool(args.params, resident=args.pool_resident or None,
                               log_path=os.path.join(project, "pool_selections.jsonl"))

    if args.arm != "reference":
        # campaign.py:262 is the only construction of a predictor in the repository, so an
        # arm is chosen by rebinding that one name. An upstream PR would make it a factory
        # read from settings; the loop itself needs no change either way.
        import ttbio_predictor as T
        import functools
        campaign.AlphaFoldDesignModel = functools.partial(
            T.TTBioAlphaFoldDesignModel, trunk="jax" if args.arm == "control" else "device",
            pool=pool, seqlog=os.path.join(project, "sequences.jsonl"))

    from bindcraft.af2 import MONOMER_POOL as _MONO_POOL
    from bindcraft.campaign import select_design_and_validation_models as _select
    # Asked AFTER the MULTIMER_POOL rebind above, so it reports what this arm really runs.
    _resolved = _select(settings, campaign.MULTIMER_POOL, _MONO_POOL)

    mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
    stamp = {"arm": args.arm, "seed": args.seed, "trajectories": args.trajectories,
             # The EFFECTIVE bucket, read back off the settings the campaign gets.
             # This used to record args.bucket, which is the flag: three device
             # trajectories and five reference streams all stamped
             # length_bucket_size 1 while the override was missing from the
             # overrides list entirely and every run used BindCraft 2's default 32.
             "length_bucket_size": campaign_length_bucket(settings),
             "length_bucket_flag": args.bucket,
             "multimer_pool": bool(args.multimer_pool),
             "levers": not args.no_levers,
             # The RESOLVED pool, not the settings field. `design_models` in settings is
             # empty whenever the pool is the default one, because BindCraft 2 resolves it
             # later in `select_design_and_validation_models` -- so stamping the field
             # wrote `design_models: []` on the first shipped-pool run and left the
             # artifact unable to say which five checkpoints produced it.
             "design_models": list(_resolved.design_models),
             "validation_models": list(_resolved.validation_models),
             "predictor": "AlphaFoldDesignModel" if args.arm == "reference"
                          else "TTBioAlphaFoldDesignModel",
             "host": os.uname().nodename, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "stage_plan": B.stage_plan(settings),
             "threads": os.environ.get("XLA_FLAGS", ""), "project": project}
    (pathlib.Path(project) / "arm_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(json.dumps(stamp, indent=1), flush=True)

    t0 = time.time()
    if args.arm == "device":
        # The Evoformer runs on card for the WHOLE campaign: every trajectory, every
        # gradient step, every validation refold. The mask travels with each call, so a
        # new binder length per trajectory needs nothing from us.
        import afgrad as _A
        from splice import EvoformerOnDevice, evoformer_on_device
        _lv = None
        if not args.no_levers:
            import stack as _S
            _lv = _S.Levers()
        if pool is not None:
            # The pool IS the Dev the splice holds: it forwards up/down/sync/stack to whichever
            # of the five trunks BindCraft 2 picked for this step, so splice.py is unchanged.
            pool.use(pool.models[0])
            _dev = pool
        else:
            _dm, _ = _A.load_models(_A.DEFAULT_PARAMS)
            _dev = _A.Dev(_dm.to_device())
        if _lv is not None:
            _lv.arm("stack")
        evo = EvoformerOnDevice(_dev, k_evo=48)
        stamp["device_card"] = int(os.environ.get("TT_VISIBLE_DEVICES", "-1"))
        with evoformer_on_device(evo):
            count = campaign.run_campaign(settings, project, af2_weights=args.params,
                                          mpnn_weights=mpnn,
                                          max_trajectories=args.trajectories)
        stamp["device_calls"] = dict(evo.calls)
        if pool is not None:
            stamp["pool"] = pool.stamp()
    else:
        count = campaign.run_campaign(settings, project, af2_weights=args.params,
                                      mpnn_weights=mpnn,
                                      max_trajectories=args.trajectories)
    stamp.update({"trajectories_run": count, "wall_seconds": round(time.time() - t0, 1),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                  "loadavg_end": os.getloadavg()})
    (pathlib.Path(project) / "arm_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(json.dumps(stamp, indent=1), flush=True)


if __name__ == "__main__":
    main()
