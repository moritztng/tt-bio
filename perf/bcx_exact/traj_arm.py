#!/usr/bin/env python3
"""A full BindCraft 2 campaign on the shipped `examples/pdl1.json`, with `exact_training` on or off.

This is bar clause 1 of `state/bcx-exact.md`: the only test that settles whether BindCraft 2's
design loop still designs without the float64 host instrument. Everything BindCraft 2 decides
stays BindCraft 2's. The settings file is unedited, the five-model multimer design pool is the
shipped one, the stage plan, the 125 gradient steps, the recycles, the filters and the ranking
are untouched, and `campaign_predictor`'s default keeps the validation ensemble on BindCraft 2's
own JAX trunk so the instrument that grades a design is the same on both arms. Two things are
set: the campaign seed, and where the project lands.

The arm is one context manager around the campaign::

    with autograd.exact_training(False):
        campaign.run_campaign(...)

`exact_training` is a dynamic-extent stack read by `tape()` when it OPENS, so this reaches every
round and every block without an edit to `tt_bio/bindcraft2.py` (`perf/bcx_exact/OFFSWITCH.md`).

The counters decide whether the arm was really the arm. `EXACT_SOFTMAX_STATS` and
`EXACT_LAYER_NORM_STATS` are snapshotted at the start and dumped at the end, and an OFF arm whose
counters moved is a failed run, not a fast one -- reported as such rather than quietly averaged.

The counter delta is ALSO flushed to `traj_live.json` every 30 s while the campaign runs. A
witness written only at campaign end is unreadable while the arm is alive, which is exactly
when somebody needs it, and is lost outright if the run dies -- taking the whole arm-identity
record with it. This fleet has paid for that shape once already.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meter as M                                                      # noqa: E402
import bc2_state as B                                                  # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings  # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402

from tt_bio import autograd as ag                                      # noqa: E402
from tt_bio import bindcraft2 as bc2                                   # noqa: E402


def git_head():
    try:
        return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception as exc:                                           # noqa: BLE001
        return f"unknown: {exc}"


def _flush(path, blob):
    """Atomic write, so a reader never catches a half-written witness."""
    tmp = pathlib.Path(f"{path}.tmp")
    tmp.write_text(json.dumps(blob, indent=1))
    tmp.replace(path)


def witness(project, exact, before, stop):
    """Flush the running counter delta until `stop` is set. Daemon, read-only on the counters."""
    path = pathlib.Path(project) / "traj_live.json"
    while True:
        counters = moved(before, snap())
        entered = sum(v for op in counters for k, v in counters[op].items() if k in ("verb", "raw"))
        _flush(path, {"utc": time.strftime("%FT%TZ", time.gmtime()), "exact_training": exact,
                      "exact_counters": counters, "entered": entered,
                      "arm_held": (entered == 0) if not exact else (entered > 0),
                      "loadavg": os.getloadavg()})
        if stop.wait(30):
            return


def snap():
    return {"softmax": dict(ag.EXACT_SOFTMAX_STATS), "layer_norm": dict(ag.EXACT_LAYER_NORM_STATS)}


def moved(before, after):
    return {op: {k: after[op][k] - before[op][k] for k in before[op]} for op in before}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", choices=["on", "off"], required=True,
                    help="on reproduces what origin/main runs today; off is the arm under test")
    ap.add_argument("--trajectories", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--settings", default=None,
                    help="defaults to the SHIPPED examples/pdl1.json, which is what the bar reads")
    ap.add_argument("--resident", type=int, default=1,
                    help="trunks kept on card. Five is about 910 MB and brought an allocator "
                         "refusal forward at n=288 that 1 ran past")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    settings_path = args.settings or os.path.join(B.BC2, "examples", "pdl1.json")
    settings = cleaned_campaign_settings(read_settings(
        settings_path,
        parse_setting_overrides([f"campaign_seed={args.seed}",
                                 f"max_trajectories={args.trajectories}",
                                 f"project_folder={project}"])))

    exact = args.exact == "on"
    M.CLOCK = M.Clock(5.0)
    M.CLOCK.start()
    before = snap()
    stamp = {"exact_training": exact, "exact_training_ops": list(ag.exact_training_ops()),
             "seed": args.seed, "trajectories": args.trajectories, "settings": settings_path,
             "settings_edited": bool(args.settings), "resident": args.resident,
             "length_bucket_size": campaign_length_bucket(settings),
             "multimer_pool": list(campaign.MULTIMER_POOL),
             "host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
             "omp": os.environ.get("OMP_NUM_THREADS"), "nproc": os.cpu_count(),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "project": project}
    (pathlib.Path(project) / "traj_stamp.json").write_text(json.dumps(stamp, indent=1))
    print(json.dumps(stamp, indent=1), flush=True)

    mpnn = os.path.join(B.BC2, "bindcraft", "weights", "proteinmpnn", "weights_neutral")
    t0, count, failed = time.time(), None, None
    stop = threading.Event()
    threading.Thread(target=witness, args=(project, exact, before, stop), daemon=True).start()
    try:
        with bc2.campaign_predictor(checkpoints=args.params, resident=args.resident) as build:
            with ag.exact_training(exact):
                count = campaign.run_campaign(settings, project, af2_weights=args.params,
                                              mpnn_weights=mpnn,
                                              max_trajectories=args.trajectories)
            stamp["device_calls"] = dict(build.evoformer.calls) if build.evoformer else None
            stamp["host_folds"] = dict(build.evoformer.host_folds) if build.evoformer else None
            stamp["pool_absent"] = dict(build.pool.absent) if build.pool else None
            stamp["pool_selections"] = dict(build.pool.selections) if build.pool else None
    except BaseException as exc:                                       # noqa: BLE001
        failed = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stop.set()
        M.CLOCK.stop()
        clk = sorted(c for _t, c, _ld in M.CLOCK.samples)
        counters = moved(before, snap())
        entered = sum(v for op in counters for k, v in counters[op].items()
                      if k in ("verb", "raw"))
        stamp.update({
            "trajectories_run": count, "failed": failed,
            "wall_seconds": round(time.time() - t0, 1),
            "exact_counters": counters,
            # The arm's own identity check. An OFF arm that entered the exact path measured
            # something else; an ON arm that did not was never the instrument's arm.
            "arm_held": (entered == 0) if not exact else (entered > 0),
            "aiclk_n": len(clk), "aiclk_min": clk[0] if clk else None,
            "aiclk_med": clk[len(clk) // 2] if clk else None,
            "aiclk_max": clk[-1] if clk else None,
            "loadavg_end": os.getloadavg(),
            "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
        _flush(pathlib.Path(project) / "traj_stamp.json", stamp)
        print(json.dumps(stamp, indent=1), flush=True)


if __name__ == "__main__":
    main()
