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
import bindcraft.campaign as campaign                                  # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402

MONOMER = ("model_1_ptm", "model_2_ptm")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["reference", "device"], default="reference")
    ap.add_argument("--trajectories", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", default=None)
    ap.add_argument("--settings", default=None)
    ap.add_argument("--bucket", type=int, default=1,
                    help="length_bucket_size. 1 leaves the complex unpadded, which is\nthe only regime tt-bio's AF2 trunk serves: it asserts an all-ones mask\n(af2.py:385, :509) and BindCraft 2's design-chain padding is masked.")
    args = ap.parse_args()

    project = args.out or str(HERE / "runs" / f"{args.arm}_seed{args.seed}")
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", f"max_trajectories={args.trajectories}",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={project}"]
    settings = cleaned_campaign_settings(
        read_settings(args.settings or os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))

    # Both arms on the monomer checkpoints -- see the module docstring.
    campaign.MULTIMER_POOL = MONOMER

    if args.arm == "device":
        raise SystemExit("the device arm needs the tt-bio predictor class; run --arm reference")

    mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
    stamp = {"arm": args.arm, "seed": args.seed, "trajectories": args.trajectories,
             "length_bucket_size": args.bucket,
             "host": os.uname().nodename, "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "stage_plan": B.stage_plan(settings),
             "threads": os.environ.get("XLA_FLAGS", ""), "project": project}
    (pathlib.Path(project) / "arm_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(json.dumps(stamp, indent=1), flush=True)

    t0 = time.time()
    count = campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=mpnn, max_trajectories=args.trajectories)
    stamp.update({"trajectories_run": count, "wall_seconds": round(time.time() - t0, 1),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime()),
                  "loadavg_end": os.getloadavg()})
    (pathlib.Path(project) / "arm_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(json.dumps(stamp, indent=1), flush=True)


if __name__ == "__main__":
    main()
