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
from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL                  # noqa: E402
from bindcraft.settings import (parse_setting_overrides, read_settings,  # noqa: E402
                                select_design_and_validation_models)
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
    ap.add_argument("--bucket", type=int, default=1,
                    help="length_bucket_size override; 0 leaves BindCraft 2's own "
                         "default of 32. It used to have to be 1 because the trunk "
                         "refused a masked fold; all three mask sites are in af2.py "
                         "now, so 32 runs and is the lab's configuration.")
    ap.add_argument("--shipped", action="store_true",
                    help="leave BindCraft 2's own model pool alone. The monomer pin below "
                         "makes the two arms comparable to each other; it also makes neither "
                         "comparable to the lab's own figure, which is the question leg 2 of "
                         "bcx-repin asks. Reference-only: tt-bio's trunk is monomer model_1_ptm.")
    args = ap.parse_args()
    if args.shipped and args.arm != "reference":
        ap.error("--shipped is reference-only; tt-bio's AF2 trunk is monomer model_1_ptm")

    project = args.out or str(HERE / "runs" / f"{args.arm}_seed{args.seed}")
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", f"max_trajectories={args.trajectories}",
                 f"project_folder={project}"]
    if not args.shipped:
        overrides[2:2] = ["validation_model=monomer", 'design_models=["model_1_ptm"]',
                          'validation_models=["model_2_ptm"]']
    if args.bucket:
        overrides.append(f"length_bucket_size={args.bucket}")
    settings = cleaned_campaign_settings(
        read_settings(args.settings or os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    _selected = select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)

    # Both arms on the monomer checkpoints -- see the module docstring.
    if not args.shipped:
        campaign.MULTIMER_POOL = MONOMER

    if args.arm != "reference":
        # campaign.py:262 is the only construction of a predictor in the repository, so an
        # arm is chosen by rebinding that one name. An upstream PR would make it a factory
        # read from settings; the loop itself needs no change either way.
        import ttbio_predictor as T
        import functools
        campaign.AlphaFoldDesignModel = functools.partial(
            T.TTBioAlphaFoldDesignModel, trunk="jax" if args.arm == "control" else "device")

    mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
    stamp = {"arm": args.arm, "seed": args.seed, "trajectories": args.trajectories,
             # The EFFECTIVE bucket, read back off the settings the campaign gets.
             # This used to record args.bucket, which is the flag: three device
             # trajectories and five reference streams all stamped
             # length_bucket_size 1 while the override was missing from the
             # overrides list entirely and every run used BindCraft 2's default 32.
             "length_bucket_size": campaign_length_bucket(settings),
             "length_bucket_flag": args.bucket,
             "shipped_model_pool": bool(args.shipped),
             # The pools BindCraft 2's own resolver gives, not the ones the flags asked
             # for. An accepted-binder count read without them is the conflation
             # state/bcx/MODELPOOL.md was written about.
             "resolved_design_models": list(_selected.design_models),
             "resolved_validation_models": list(_selected.validation_models),
             "settings_file": args.settings or os.path.join(B.BC2, "examples", "pdl1.json"),
             "bc2": B.BC2,
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
        import afgrad as _A, stack as _S
        from splice import EvoformerOnDevice, evoformer_on_device
        _lv = _S.Levers()
        _dm, _ = _A.load_models(_A.DEFAULT_PARAMS)
        _dev = _A.Dev(_dm.to_device())
        _lv.arm("stack")
        evo = EvoformerOnDevice(_dev, k_evo=48)
        stamp["device_card"] = int(os.environ.get("TT_VISIBLE_DEVICES", "-1"))
        with evoformer_on_device(evo):
            count = campaign.run_campaign(settings, project, af2_weights=args.params,
                                          mpnn_weights=mpnn,
                                          max_trajectories=args.trajectories)
        stamp["device_calls"] = dict(evo.calls)
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
