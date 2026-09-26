#!/usr/bin/env python3
"""Does `bindcraft2.predictor(exact=...)` actually reach the float64 instrument?

`perf/bcx_exact/round_ab.py` proved what `exact_training` COSTS by flipping the context manager
itself inside its own `sequence_gradients` wrapper. That is a different claim from the one the
shipped parameter makes. A parameter can be wired and inert -- this campaign has already paid
for that once, with an extra-MSA ratio measured on a `splice.py` that no longer existed -- so
the parameter gets its own runtime check, through the entry point a caller actually has.

The route under test is the public one and nothing else:

    with bindcraft2.campaign_predictor(checkpoints=..., exact=<arm>) as build:
        campaign.run_campaign(...)

No `exact_training` call appears in this file outside the assertions. The arm is the keyword
argument. The verdict is the live counters around one real BindCraft 2 gradient round:

    exact=False   EXACT_SOFTMAX_STATS and EXACT_LAYER_NORM_STATS move by EXACTLY zero, and
                  `exact_training_ops()` reads `()` inside the scope
    exact=True    both counters move, and `exact_training_ops()` reads the armed tuple

An OFF arm whose counters moved is a failed run, not a fast one, and this exits non-zero on it.
The wall per round is recorded too, but it is one round per arm in separate campaigns, so it is
a sanity reading beside `round_ab.json`'s interleaved 24.87x, not a replacement for it.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT), str(_ROOT / "perf" / "bcx_stack")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meter as M                                                      # noqa: E402
import bc2_state as B                                                  # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
import bindcraft.trajectory as trajectory                              # noqa: E402
import bindcraft.sequence_optimization as seqopt                       # noqa: E402
from bindcraft.af2 import campaign_length_bucket                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from run_round import MONOMER, git_head                                # noqa: E402

from tt_bio import autograd as ag                                      # noqa: E402

ROUNDS = []


def snap():
    return {"sm": dict(ag.EXACT_SOFTMAX_STATS), "ln": dict(ag.EXACT_LAYER_NORM_STATS)}


def delta(a, b):
    return {f"sm_{k}": b["sm"][k] - a["sm"][k] for k in a["sm"]} | \
           {f"ln_{k}": b["ln"][k] - a["ln"][k] for k in a["ln"]}


def wrap_round(cls):
    """Counter delta and wall around each round, OUTSIDE the meter's own wrapper.

    The meter raises `StopAfterRounds` from inside, so the delta is recorded in `finally` and
    the last round is not lost to the stop.
    """
    sg = cls.sequence_gradients

    def sequence_gradients(self, *a, **kw):
        a0, t0 = snap(), time.time()
        armed = list(ag.exact_training_ops())
        try:
            return sg(self, *a, **kw)
        finally:
            ROUNDS.append({"seconds": round(time.time() - t0, 3), "armed_ops": armed,
                           **delta(a0, snap())})
    cls.sequence_gradients = sequence_gradients


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exact", choices=("on", "off"), required=True)
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    exact = args.exact == "on"

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    # The same draw and the same pool as `round_ab.py`, so the wall lands on its axis.
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={project}",
                 "compile_next_length=false"]
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(B.BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))
    campaign.MULTIMER_POOL = MONOMER
    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
             "arm": args.exact, "seed": args.seed, "rounds_requested": args.rounds,
             "route": "bindcraft2.campaign_predictor(exact=%r)" % exact,
             "ops_armed_before_scope": list(ag.exact_training_ops()),
             "length_bucket_size": campaign_length_bucket(settings),
             "omp": os.environ.get("OMP_NUM_THREADS"),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    from tt_bio import bindcraft2 as bc2
    cls = bc2.design_model_class()
    mt = M.Meter(args.rounds)
    M.install(mt, bc2, cls, trajectory, seqopt)
    wrap_round(cls)

    t0, stopped, evo, inside = time.time(), None, None, None
    try:
        with bc2.campaign_predictor(checkpoints=args.params, exact=exact) as build:
            evo = build.evoformer
            inside = {"build_exact": build.exact,
                      "ops_armed_inside_scope": list(ag.exact_training_ops())}
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                            "proteinmpnn", "weights_neutral"),
                                  max_trajectories=1)
    except M.StopAfterRounds as stop:
        stopped = str(stop)
    finally:
        M.CLOCK.stop()

    after = list(ag.exact_training_ops())
    samples = [c for _, c, _ in M.CLOCK.samples]
    # The meter raises StopAfterRounds from the NEXT rounds entry, which this wrapper
    # also sees, at a wall of microseconds. That entry ran no gradient and is not a round.
    counted = [r for r in ROUNDS if r["seconds"] > 1.0]
    moved = {k: sum(r[k] for r in counted) for k in
             ("sm_verb", "sm_raw", "ln_verb", "ln_raw", "ln_bw", "ln_elements")}
    checks = {
        "ops_empty_inside_off_scope": (inside or {}).get("ops_armed_inside_scope") == []
        if not exact else None,
        "ops_armed_inside_on_scope": bool((inside or {}).get("ops_armed_inside_scope"))
        if exact else None,
        "scope_restored_after": after == list(ag.EXACT_TRAINING_OPS),
        "build_exact_matches": (inside or {}).get("build_exact") is exact,
        "rounds_ran": len(counted),
        "counters_exactly_zero": all(v == 0 for v in moved.values()),
        "counters_moved": moved["sm_verb"] + moved["ln_verb"] > 0}
    stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                  "evo_calls": dict(evo.calls) if evo else None,
                  "aiclk_during": {"median": sorted(samples)[len(samples) // 2] if samples
                                   else None, "min": min(samples) if samples else None,
                                   "n": len(samples)},
                  "loadavg_end": os.getloadavg(),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
    out = {"stamp": stamp, "inside_scope": inside, "ops_after_scope": after,
           "rounds": ROUNDS, "counter_totals": moved, "checks": checks}
    pathlib.Path(project, f"reach_public_{args.exact}.json").write_text(
        json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str), flush=True)

    if not counted:
        raise SystemExit("no gradient round ran: the check measured nothing")
    if exact and not checks["counters_moved"]:
        raise SystemExit("exact=True did not reach the float64 path: the parameter is inert")
    if not exact and not checks["counters_exactly_zero"]:
        raise SystemExit("exact=False still entered the float64 path: the parameter is inert")
    if not checks["build_exact_matches"] or not checks["scope_restored_after"]:
        raise SystemExit("the scope did not open or did not close as the parameter says")


if __name__ == "__main__":
    main()
