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
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import bindcraft.campaign as campaign                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from tt_bio import bindcraft2                                          # noqa: E402

MONOMER = ("model_1_ptm", "model_2_ptm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["reference", "control", "device"], default="reference",
                    help="reference: BindCraft 2's own AlphaFoldDesignModel, untouched. "
                         "control: tt_bio's design model with trunk='jax' -- our class, "
                         "BindCraft 2's trunk, so a device result is compared against the same "
                         "call path. device: trunk on card.")
    ap.add_argument("--trajectories", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", default=None)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--binder-length", type=int, default=0,
                    help="pin the drawn binder length, so a size can be held while something "
                         "else varies; 0 leaves BindCraft 2 to draw from its own range. A "
                         "paired comparison needs matched draws and cannot run without it.")
    ap.add_argument("--no-levers", action="store_true",
                    help="run the pre-lever control: install perf/bcx_stack's Levers with every "
                         "switch cleared, so _via2d, bmm_program_config, add_grad_slice, the "
                         "relayout add_grad and the two head verbs route to the implementations "
                         "they replaced. The levers are SHIPPED defaults in tt_bio/autograd.py, "
                         "so the default arm here is levers on and uncounted; this flag is the "
                         "control every BCX speed ratio divides by. Inert on the reference arm, "
                         "which never enters tt_bio.")
    ap.add_argument("--shipped", action="store_true",
                    help="leave BindCraft 2's own five-model multimer_v3 pool alone. The "
                         "monomer pin makes the arms comparable to each other and makes none "
                         "of them comparable to the lab's own figure, which is the only "
                         "question this flag exists to ask. Reference-only: tt-bio's AF2 trunk "
                         "is monomer model_1_ptm. Off by default, and off is byte-identical to "
                         "every arm this campaign has measured.")
    ap.add_argument("--bucket", type=int, default=1,
                    help="length_bucket_size override; 0 leaves BindCraft 2's own "
                         "default of 32. It used to have to be 1 because the trunk "
                         "refused a masked fold; all three mask sites are in af2.py "
                         "now, so 32 runs and is the lab's configuration.")
    args = ap.parse_args()
    if args.shipped and args.arm != "reference":
        ap.error("--shipped is reference-only; tt-bio's AF2 trunk is monomer model_1_ptm")

    project = args.out or str(HERE / "runs" / f"{args.arm}_seed{args.seed}")
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", f"max_trajectories={args.trajectories}",
                 f"project_folder={project}"]
    if not args.shipped:
        # The monomer pin, unchanged and still the default. Every number this campaign has
        # published was measured with these three lines in place.
        overrides[2:2] = ["validation_model=monomer", 'design_models=["model_1_ptm"]',
                          'validation_models=["model_2_ptm"]']
    if args.binder_length:
        overrides.append(f"binder_lengths=[{args.binder_length}]")
    if args.bucket:
        overrides.append(f"length_bucket_size={args.bucket}")
    settings = cleaned_campaign_settings(
        read_settings(args.settings or os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    # Both arms on the monomer checkpoints -- see the module docstring.
    if not args.shipped:
        campaign.MULTIMER_POOL = MONOMER

    mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
    stamp = {"arm": args.arm, "seed": args.seed, "trajectories": args.trajectories,
             # The EFFECTIVE bucket, read back off the settings the campaign gets.
             # This used to record args.bucket, which is the flag: three device
             # trajectories and five reference streams all stamped
             # length_bucket_size 1 while the override was missing from the
             # overrides list entirely and every run used BindCraft 2's default 32.
             "length_bucket_size": campaign_length_bucket(settings),
             "length_bucket_flag": args.bucket,
             # The EFFECTIVE draw and lever state, both read back rather than echoed:
             # `binder_lengths` reads back as BindCraft 2's own range, [60, 180] on
             # pdl1.json, when the flag is 0, and as the single pinned length when it is not.
             "binder_lengths": settings.get("binder_lengths"),
             "levers": "off" if args.no_levers else "shipped",
             "shipped_model_pool": bool(args.shipped),
             "predictor": "AlphaFoldDesignModel" if args.arm == "reference"
                          else "TenstorrentAlphaFoldDesignModel",
             "host": os.uname().nodename, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "stage_plan": B.stage_plan(settings),
             "threads": os.environ.get("XLA_FLAGS", ""), "project": project}
    (pathlib.Path(project) / "arm_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(json.dumps(stamp, indent=1), flush=True)

    def run():
        return campaign.run_campaign(settings, project, af2_weights=args.params,
                                     mpnn_weights=mpnn,
                                     max_trajectories=args.trajectories)

    t0 = time.time()
    if args.arm == "reference":
        count = run()
    else:
        # `campaign_predictor` rebinds `AlphaFoldDesignModel` at campaign.py's single
        # construction site and, for the device arm, puts the Evoformer on card for the
        # WHOLE campaign: every trajectory, every gradient step, every validation refold.
        # The mask travels with each call, so a new binder length per trajectory needs
        # nothing from us.
        #
        # validation="device" keeps the validation ensemble on card too, which is the arm
        # every number in state/bcx-predictor.md was measured on. tt_bio's own default
        # holds validation on BindCraft 2's JAX trunk, which is the right default for a
        # campaign whose accepted count is the result and the wrong one for a comparison
        # whose subject is the card. It is inert on the control arm.
        if args.no_levers:
            # Install BEFORE the first taped call -- taped_ttnn caches a verb's tape entry on
            # first use, so a later install would leave the already-cached verbs on the new
            # path. Clearing every switch is what makes this the control: Levers() constructed
            # with its defaults routes to the same implementations the shipped arm uses and
            # only adds counters.
            sys.path.insert(0, str(_ROOT / "perf" / "bcx_stack"))
            import stack as _S                                          # noqa: E402
            _levers = _S.Levers()
            _levers.mm2d = _levers.bmm = _levers.heads = False
            _levers.bwd = set()
        with bindcraft2.campaign_predictor(
                trunk="jax" if args.arm == "control" else "device",
                validation="device", checkpoints=args.params) as build:
            if build.evoformer is not None:
                stamp["device_card"] = int(os.environ.get("TT_VISIBLE_DEVICES", "-1"))
            count = run()
            if build.evoformer is not None:
                stamp["device_calls"] = dict(build.evoformer.calls)
                # Which checkpoints actually folded on card. The pool falls back to the
                # host per model FAMILY when it cannot supply every checkpoint that family
                # draws from, so a run that quietly folded on the host says so here rather
                # than being read as a device number.
                stamp["device_checkpoints"] = sorted(build.pool.paths)
                stamp["host_checkpoints"] = sorted(build.pool.absent)
                stamp["host_folds"] = dict(build.evoformer.host_folds)
    stamp.update({"trajectories_run": count, "wall_seconds": round(time.time() - t0, 1),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                  "loadavg_end": os.getloadavg()})
    (pathlib.Path(project) / "arm_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(json.dumps(stamp, indent=1), flush=True)


if __name__ == "__main__":
    main()
