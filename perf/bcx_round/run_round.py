#!/usr/bin/env python3
"""Run BindCraft 2's own campaign on the card and time N consecutive gradient rounds.

Same arm as `perf/bcx_predictor/run_arm.py --arm device`: BindCraft 2's `campaign.py`
drives, our predictor class is rebound at its single construction site, the Evoformer runs
on the card for every call. The only addition is `perf/bcx_round/meter.py`, which
timestamps the seams. `--rounds` stops collection after N rounds have run in full; it does
not shorten a round, reduce recycles or skip a stage.
"""
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(_ROOT / "perf" / "bcx_predictor"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import meter as M                                                      # noqa: E402
import bc2_state as B                                                  # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
import bindcraft.trajectory as trajectory                              # noqa: E402
import bindcraft.sequence_optimization as seqopt                       # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from tt_bio import bindcraft2                                          # noqa: E402

MONOMER = ("model_1_ptm", "model_2_ptm")


def git_head():
    return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=14)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--bucket", type=int, default=0,
                    help="length_bucket_size override; 0 leaves BindCraft 2's own 32")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    project = args.out or str(HERE / "runs" / f"round_seed{args.seed}")
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={project}"]
    if args.bucket:
        overrides.append(f"length_bucket_size={args.bucket}")
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))
    campaign.MULTIMER_POOL = MONOMER

    node = None
    M.CLOCK = M.Clock(1.0)
    node = M.CLOCK.path
    M.CLOCK.start()

    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "sysfs": node, "commit": git_head(),
             "seed": args.seed, "rounds_requested": args.rounds,
             "length_bucket_size": campaign_length_bucket(settings),
             "stage_plan": B.stage_plan(settings),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(),
             "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    # `meter.install` patches EvoformerOnDevice's three seams and the predictor's two entry
    # points, so it takes the module and the class rather than instances. Both now come from
    # tt_bio's shipped surface.
    mt = M.Meter(args.rounds)
    M.install(mt, bindcraft2, bindcraft2.design_model_class(), trajectory, seqopt)

    # No stack.Levers here, unlike run_arm.py once did. Its three lever arms are all ON in
    # this tree already (autograd.TRIATT_BMM_CONFIG is True and ARMS["stack"] is (1, 1, 1)),
    # so arming changes no kernel, and its per-op counting wrappers are Python on the host
    # path this row is trying to measure.
    #
    # The trunks load lazily, on the first fold that reaches one, so there is no load_models
    # seam to time here any more. `meter` sees that cost inside the first device call.
    t0 = time.time()
    stopped = None
    evo = None
    try:
        with bindcraft2.campaign_predictor(trunk="device", validation="device",
                                           checkpoints=args.params) as build:
            evo = build.evoformer
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                            "proteinmpnn", "weights_neutral"),
                                  max_trajectories=1)
    except M.StopAfterRounds as stop:
        stopped = str(stop)
    finally:
        M.CLOCK.stop()
        stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                      "device_calls": dict(evo.calls) if evo else None,
                      "host_folds": dict(evo.host_folds) if evo else None,
                      "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        out = pathlib.Path(project) / "round_events.json"
        M.dump(str(out), stamp)
        print(json.dumps(stamp, indent=1), flush=True)
        print(f"events -> {out}", flush=True)


if __name__ == "__main__":
    main()
