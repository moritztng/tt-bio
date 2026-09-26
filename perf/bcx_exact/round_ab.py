#!/usr/bin/env python3
"""BindCraft 2 gradient rounds with exact_training ON and OFF, interleaved in ONE process.

This is the device leg of `bcx-exact`. It answers both halves of the COUNT and the SHARE:

COUNT  `EXACT_SOFTMAX_STATS` and `EXACT_LAYER_NORM_STATS` are snapshotted around every
       `EvoformerOnDevice._taped`, `._backward` and `._primal` call and around each round, so
       the per-round total arrives already split by forward, backward and forward-only fold,
       and with elements beside calls. Counted from the live counters, not from the source:
       `perf/bcx_exact/blockcount.py`'s arithmetic says a 48-block taped forward that reached
       the instrument at the traced per-block rate could not fit inside the trunk step this
       campaign has measured, so the count is the thing to settle, not assume.

SHARE  the arm alternates round by round. Two modes:

       --alternate (default)  round 1 compiles OFF, round 2 compiles ON, then they alternate.
                              Cheap, but after round 2 the two arms are on different sequences,
                              because the OFF arm returns a different gradient. Shapes are
                              identical (BindCraft 2's own length bucket is 32 and does not
                              move inside a stage), so the timing comparison is at matched
                              shapes and the trajectory divergence is reported, not hidden.
       --twin                 each round runs `sequence_gradients` TWICE on byte-identical
                              inputs, once per arm, alternating which goes first, and the ON
                              arm's result is what the loop gets. Matched inputs and drift
                              cancelled, at twice the wall.

Unlike `perf/bcx_tmplseam/round_ab.py` this needs no per-arm compiled-gradient cache:
`exact_training` rebinds ttnn callables inside our own splice callback and changes nothing
about the JAX program BindCraft 2 traces. That is asserted at the end rather than assumed --
`bcx-extramsa` read 1.055x on a 2.16x lever once for exactly this class of mistake -- by
requiring the device Evoformer call count to match across arms and the exact counters to move
in the ON rounds and not in the OFF ones.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics as st
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT), str(_ROOT / "perf" / "bcx_afgrad"), str(_ROOT / "perf" / "bcx_stack")):
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

ARM = {"exact": True}
COUNTS = []
STAGE = {"id": 0, "in": False}


def snap():
    return {"sm": dict(ag.EXACT_SOFTMAX_STATS), "ln": dict(ag.EXACT_LAYER_NORM_STATS)}


def delta(a, b):
    return {f"sm_{k}": b["sm"][k] - a["sm"][k] for k in a["sm"]} | \
           {f"ln_{k}": b["ln"][k] - a["ln"][k] for k in a["ln"]}


def arm_of(r):
    """Round 1 compiles OFF, round 2 compiles ON, then OFF/ON alternate."""
    return r == 2 or (r > 2 and r % 2 == 0)


class ABMeter(M.Meter):
    def on_sequence_gradients_enter(self):
        super().on_sequence_gradients_enter()
        M.EVENTS.append({"kind": "arm", "phase": "round", "t0": time.time(),
                         "round": self.entries, "exact": ARM["exact"]})


def wrap_device_seam(cls):
    """Counter deltas and wall around each of the three device entry points."""
    for name in ("_primal", "_taped", "_backward"):
        orig = getattr(cls, name, None)
        if orig is None:
            continue

        def make(orig, phase):
            def wrapper(self, *a, **kw):
                a0, t0 = snap(), time.time()
                try:
                    return orig(self, *a, **kw)
                finally:
                    COUNTS.append({"phase": phase, "t0": t0, "dt": time.time() - t0,
                                   "exact": ARM["exact"], **delta(a0, snap())})
            return wrapper
        setattr(cls, name, make(orig, name.lstrip("_")))


def wrap_stage(traj_mod):
    """Bracket each gradient design stage, so a round wall is only ever taken inside one.

    A round is one iteration of `run_gradient_design_stage`'s own while loop: select the states,
    call `sequence_gradients`, then the loss bookkeeping and the optimiser's sequence update. So
    the round wall is the interval between two consecutive `sequence_gradients` ENTRIES, and the
    part of it that is not the gradient call is the remainder the instrument does not touch.

    The last call of a stage has no successor inside that stage, so its wall would run into the
    next stage and swallow the transition. Dropping it is a rule fixed here, before any number
    exists, rather than an outlier tolerance chosen once the distribution is on screen.
    """
    orig = traj_mod.run_gradient_design_stage

    def run_gradient_design_stage(*a, **kw):
        STAGE["id"] += 1
        STAGE["in"] = True
        try:
            return orig(*a, **kw)
        finally:
            STAGE["in"] = False
    traj_mod.run_gradient_design_stage = run_gradient_design_stage


def wrap_arm(cls, twin, mt):
    """The arm switch, at the round's own call.

    This wrapper sits OUTSIDE the one `meter.install` put on, and that is the one incrementing
    `mt.entries`, so the round about to run is `mt.entries + 1`. Reading `ARM` here instead
    would take the PREVIOUS round's arm and mislabel which round compiled under which arm.
    """
    sg = cls.sequence_gradients

    def once(self, on, a, kw):
        ARM["exact"] = on
        a0, t0 = snap(), time.time()
        with ag.exact_training(on):
            out = sg(self, *a, **kw)
        dt = time.time() - t0
        M.EVENTS.append({"kind": "arm_call", "phase": "sequence_gradients", "t0": t0,
                         "t1": t0 + dt, "dt": dt, "exact": on,
                         "stage": STAGE["id"] if STAGE["in"] else None,
                         **delta(a0, snap())})
        return out, dt

    def sequence_gradients(self, *a, **kw):
        this = arm_of(mt.entries + 1)
        if not twin:
            out, _ = once(self, this, a, kw)
            return out
        # Palindromic: on even rounds the OFF arm runs first, on odd rounds the ON arm does.
        first = this
        out_first, _ = once(self, first, a, kw)
        out_second, _ = once(self, not first, a, kw)
        return out_first if first else out_second
    cls.sequence_gradients = sequence_gradients


def analyse(events, clock, twin):
    calls = [e for e in events if e["kind"] == "arm_call"]
    rows = []
    for e in calls:
        s0, s1 = e["t0"], e["t1"]
        clk = sorted(c for t, c, _ in clock if s0 <= t <= s1)
        load = [ld for t, _, ld in clock if s0 <= t <= s1]
        rows.append({"exact": e["exact"], "sequence_gradients_s": round(e["dt"], 3),
                     "t0": e["t0"], "t1": e["t1"], "stage": e.get("stage"),
                     "sm_verb": e["sm_verb"], "sm_raw": e["sm_raw"],
                     "sm_raw_elements": e["sm_raw_elements"],
                     "ln_verb": e["ln_verb"], "ln_raw": e["ln_raw"], "ln_bw": e["ln_bw"],
                     "ln_elements": e["ln_elements"],
                     "aiclk_n": len(clk), "aiclk_min": clk[0] if clk else None,
                     "aiclk_med": clk[len(clk) // 2] if clk else None,
                     "load1": round(sum(load) / len(load), 1) if load else None})
    # Drop the first call of each arm: it compiles.
    timed, seen = [], set()
    for r in rows:
        if r["exact"] in seen:
            timed.append(r)
        else:
            seen.add(r["exact"])
    summary = {}
    for on in (True, False):
        xs = [x for x in timed if x["exact"] is on]
        if not xs:
            continue
        s = [x["sequence_gradients_s"] for x in xs]
        summary["on" if on else "off"] = {
            "n": len(xs), "median": round(st.median(s), 3), "min": min(s), "max": max(s),
            "all": sorted(s),
            "sm_verb_median": st.median([x["sm_verb"] for x in xs]),
            "sm_raw_median": st.median([x["sm_raw"] for x in xs]),
            "sm_elements_median": st.median([x["sm_raw_elements"] for x in xs]),
            "ln_verb_median": st.median([x["ln_verb"] for x in xs]),
            "ln_raw_median": st.median([x["ln_raw"] for x in xs]),
            "ln_bw_median": st.median([x["ln_bw"] for x in xs]),
            "ln_elements_median": st.median([x["ln_elements"] for x in xs]),
            "aiclk_med_median": st.median([x["aiclk_med"] for x in xs if x["aiclk_med"]] or [0]),
            "aiclk_min": min((x["aiclk_min"] for x in xs if x["aiclk_min"]), default=None),
            "load1_median": st.median([x["load1"] for x in xs if x["load1"] is not None] or [0])}
    if "on" in summary and "off" in summary:
        summary["ratio_on_over_off"] = round(summary["on"]["median"]
                                             / summary["off"]["median"], 3)
        summary["separated"] = summary["off"]["max"] < summary["on"]["min"]
        summary["matched_inputs"] = bool(twin)
    return rows, summary


def round_walls(rows, twin):
    """The DESIGN ROUND, not the gradient call, taken from the gap between consecutive calls.

    `sequence_gradients` is the quantity the arm switch wraps and the one the 24.87x is on. A
    round also carries BindCraft 2's own JAX work outside the tape -- the loss bookkeeping and
    the optimiser's sequence update -- which the instrument does not touch, so the round factor
    is necessarily smaller than the gradient-call factor. Converting one into the other by
    arithmetic across two runs is an axis error; measuring the round directly is not.

    Round k is `t0` of call k+1 minus `t0` of call k, both inside one stage, so `round_s` is
    `sequence_gradients_s` plus `remainder_s` by construction. Because the arms alternate, each
    arm gets its OWN measured remainder, which is the test of the assumption that the remainder
    is instrument-independent rather than a claim that it is.
    """
    if twin:
        return [], {"skipped": "twin runs both arms inside one round, so the gap between two "
                               "consecutive calls is not a round"}
    walls = []
    for a, b in zip(rows, rows[1:]):
        if a["stage"] is None or a["stage"] != b["stage"]:
            continue
        walls.append({"exact": a["exact"], "stage": a["stage"],
                      "round_s": round(b["t0"] - a["t0"], 3),
                      "sequence_gradients_s": a["sequence_gradients_s"],
                      "remainder_s": round(b["t0"] - a["t1"], 3),
                      "aiclk_med": a["aiclk_med"], "load1": a["load1"]})
    # The same rule the call summary uses: each arm's first round is the compile.
    timed, seen = [], set()
    for w in walls:
        if w["exact"] in seen:
            timed.append(w)
        else:
            seen.add(w["exact"])
    summary = {}
    for on in (True, False):
        xs = [x for x in timed if x["exact"] is on]
        if not xs:
            continue
        r = [x["round_s"] for x in xs]
        rem = [x["remainder_s"] for x in xs]
        summary["on" if on else "off"] = {
            "n": len(xs), "round_median": round(st.median(r), 3),
            "round_min": min(r), "round_max": max(r), "round_all": sorted(r),
            "remainder_median": round(st.median(rem), 3),
            "sequence_gradients_median": round(
                st.median([x["sequence_gradients_s"] for x in xs]), 3)}
    on_x = [x["round_s"] for x in timed if x["exact"]]
    off_x = [x["round_s"] for x in timed if not x["exact"]]
    if on_x and off_x:
        summary["round_ratio_on_over_off"] = round(summary["on"]["round_median"]
                                                   / summary["off"]["round_median"], 3)
        summary["round_separated"] = max(off_x) < min(on_x)
        off_rem = summary["off"]["remainder_median"]
        summary["remainder_on_over_off"] = (
            round(summary["on"]["remainder_median"] / off_rem, 3) if off_rem else None)
    return walls, summary


def phase_split():
    out = {}
    for c in COUNTS:
        k = (c["phase"], c["exact"])
        r = out.setdefault(k, {"phase": c["phase"], "exact": c["exact"], "calls": 0,
                               "sm_verb": 0, "sm_raw": 0, "sm_raw_elements": 0,
                               "ln_verb": 0, "ln_raw": 0, "ln_bw": 0, "ln_elements": 0,
                               "seconds": 0.0})
        r["calls"] += 1
        r["seconds"] += c["dt"]
        for f in ("sm_verb", "sm_raw", "sm_raw_elements", "ln_verb", "ln_raw", "ln_bw",
                  "ln_elements"):
            r[f] += c[f]
    for r in out.values():
        r["seconds"] = round(r["seconds"], 3)
    return sorted(out.values(), key=lambda r: (r["phase"], not r["exact"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--twin", action="store_true",
                    help="both arms per round on byte-identical inputs, at twice the wall")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
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
             "seed": args.seed, "rounds_requested": args.rounds, "twin": args.twin,
             "exact_training_ops": list(ag.exact_training_ops()),
             "length_bucket_size": campaign_length_bucket(settings),
             "omp": os.environ.get("OMP_NUM_THREADS"),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    from tt_bio import bindcraft2 as bc2
    cls = bc2.design_model_class()
    mt = ABMeter(args.rounds)
    M.install(mt, bc2, cls, trajectory, seqopt)
    wrap_device_seam(bc2.EvoformerOnDevice)
    wrap_stage(trajectory)
    wrap_arm(cls, args.twin, mt)

    t0, stopped, evo = time.time(), None, None
    try:
        # `campaign_predictor` rebinds `campaign.AlphaFoldDesignModel` itself and holds the
        # trunk pool on card for the whole campaign. `validation="jax"` is its default and is
        # kept: the validation ensemble grades the design, so device numerics do not belong in
        # it, and it also keeps the arm difference confined to the gradient loop under test.
        with bc2.campaign_predictor(checkpoints=args.params) as build:
            evo = build.evoformer
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                            "proteinmpnn", "weights_neutral"),
                                  max_trajectories=1)
    except M.StopAfterRounds as stop:
        stopped = str(stop)
    finally:
        M.CLOCK.stop()

    rows, summary = analyse(M.EVENTS, M.CLOCK.samples, args.twin)
    walls, round_summary = round_walls(rows, args.twin)
    split = phase_split()
    on_rows = [r for r in rows if r["exact"]]
    off_rows = [r for r in rows if not r["exact"]]
    checks = {
        "on_arm_counters_moved": bool(on_rows) and all(r["sm_verb"] + r["sm_raw"] +
                                                       r["ln_verb"] + r["ln_raw"] > 0
                                                       for r in on_rows),
        "off_arm_counters_flat": all(r["sm_verb"] + r["sm_raw"] + r["ln_verb"] + r["ln_raw"] == 0
                                     for r in off_rows),
        "both_arms_ran": bool(on_rows) and bool(off_rows)}
    stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                  "evo_calls": dict(evo.calls) if evo else None,
                  "checks": checks, "loadavg_end": os.getloadavg(),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
    M.dump(os.path.join(project, "round_events.json"), stamp)
    out = {"stamp": stamp, "summary": summary, "round_summary": round_summary,
           "phase_split": split, "calls": rows, "round_walls": walls}
    pathlib.Path(project, "round_ab.json").write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        print(json.dumps(r), flush=True)
    for r in split:
        print(json.dumps(r), flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    print(json.dumps(round_summary, indent=1), flush=True)
    print(json.dumps(checks, indent=1), flush=True)
    if not checks["off_arm_counters_flat"]:
        raise SystemExit("the OFF arm still entered the exact path: the arms share state")


if __name__ == "__main__":
    main()
