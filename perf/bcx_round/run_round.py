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
    ap.add_argument("--exact", type=int, default=1,
                    help="predictor(exact=...); 1 is what origin/main defaults to")
    ap.add_argument("--binder", type=int, default=0,
                    help="pin binder_lengths so n is fixed; 0 leaves pdl1.json's own 60-180 draw")
    ap.add_argument("--extra-msa", dest="extra_msa", type=int, default=0,
                    help="predictor(extra_msa=...); 1 runs BindCraft 2's 4-block extra-MSA "
                         "stack on card instead of in JAX. Off is origin/main's default. "
                         "The stack is 13.77 s of the round's 20.831 host seconds "
                         "(state/perf10/bcx-HOSTMAP.md), so this is the campaign's "
                         "largest single lever")
    ap.add_argument("--shipped", action="store_true",
                    help="leave pdl1.json's own five multimer_v3 design models in place "
                         "instead of pinning one monomer trunk")
    args = ap.parse_args()

    project = args.out or str(HERE / "runs" / f"round_seed{args.seed}")
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)

    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 f"project_folder={project}"]
    if not args.shipped:
        overrides += ["validation_model=monomer", 'design_models=["model_1_ptm"]',
                      'validation_models=["model_2_ptm"]']
    if args.bucket:
        overrides.append(f"length_bucket_size={args.bucket}")
    if args.binder:
        overrides.append(f"binder_lengths=[{args.binder},{args.binder}]")
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))
    if not args.shipped:
        campaign.MULTIMER_POOL = MONOMER

    node = None
    M.CLOCK = M.Clock(1.0)
    node = M.CLOCK.path
    M.CLOCK.start()

    import tt_bio
    from bindcraft.settings import select_design_and_validation_models
    from bindcraft.af2 import MONOMER_POOL
    try:
        # Whatever BindCraft 2's own resolver returns, printed as it comes. It is a
        # `PredictionModelSelection`, not a pair, and unpacking it was the first thing this
        # stamp got wrong.
        pool = repr(select_design_and_validation_models(
            settings, campaign.MULTIMER_POOL, MONOMER_POOL))
    except Exception as exc:      # a pool BindCraft 2's own loader refuses is itself a finding
        pool = f"REFUSED: {exc}"

    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "tt_bio_file": tt_bio.__file__, "exact": bool(args.exact),
             "extra_msa_on_device": bool(args.extra_msa),
             "shipped_pool": bool(args.shipped), "binder_pinned": args.binder,
             "model_pool": pool,
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
    out = pathlib.Path(project) / "round_events.json"
    M.DUMP = (str(out), stamp)
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
    extra = None
    try:
        with bindcraft2.campaign_predictor(trunk="device", validation="device",
                                           checkpoints=args.params,
                                           exact=bool(args.exact),
                                           extra_msa=bool(args.extra_msa)) as build:
            evo = build.evoformer
            extra = build.extra_msa
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                            "proteinmpnn", "weights_neutral"),
                                  max_trajectories=1)
    except M.StopAfterRounds as stop:
        stopped = str(stop)
    finally:
        M.CLOCK.stop()
        from tt_bio import autograd
        stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                      "exact_softmax_stats": dict(autograd.EXACT_SOFTMAX_STATS),
                      "exact_layer_norm_stats": dict(autograd.EXACT_LAYER_NORM_STATS),
                      "device_calls": dict(evo.calls) if evo else None,
                      # An extra-MSA swap that never fires costs nothing and reads as a clean
                      # 0 % result. `calls` proves the on-card path ran; `swapped` is the block
                      # count it replaced; `mask_seen` is the guard's own record of what the
                      # extra_msa_mask actually carried.
                      "extra_msa_calls": dict(extra.calls) if extra else None,
                      "extra_msa_swapped": list(extra.swapped) if extra else None,
                      "extra_msa_mask_seen": dict(extra.mask_seen) if extra else None,
                      "host_folds": dict(evo.host_folds) if evo else None,
                      "loadavg_end": os.getloadavg(),
                      "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        stamp["state_shape"] = dict(M.STATE)
        M.dump(str(out), stamp)
        print(json.dumps(stamp, indent=1), flush=True)
        print(f"events -> {out}", flush=True)


if __name__ == "__main__":
    main()
