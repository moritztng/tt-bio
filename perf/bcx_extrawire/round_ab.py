#!/usr/bin/env python3
"""BindCraft 2 gradient rounds with the extra-MSA swap OFF and ON, interleaved in one process.

This is `bcx-extramsa`'s round A/B moved onto the surface we ship. That one measured 2.16x
against `perf/bcx_predictor/splice.py`, a harness-local file `bcx-backend` promoted into
`tt_bio/bindcraft2.py` and deleted, so its number does not carry over: `229e945b5` re-wrote the
lever onto the shipped file rather than picking it, and a rewrite onto a different file is a
different change. Everything here drives `tt_bio.bindcraft2.predictor(extra_msa=...)`, the
argument a user would pass.

One card, one process, seed 100, BindCraft 2's `campaign.py` driving. The Evoformer runs on the
card in both arms. The single thing that changes between rounds is whether `modules.py`'s
extra-MSA `layer_stack` is BindCraft 2's JAX or `bindcraft2.ExtraMsaOnDevice`.

Two traps this harness exists to avoid, both of which have already produced a wrong number on
this campaign:

  Shared traces. `RunModel.apply` is a `jax.jit` built once per runner
  (`af/alphafold/model/model.py:96`) and JAX keys its jaxpr on input shapes, so a second outer
  program that calls the same runner reuses the first one's trace and the layer_stack factory is
  never asked again. `bcx-extramsa`'s first A/B shared the runners, traced the extra-MSA stack
  once in the OFF arm, ran JAX in every ON round and read 1.055x. Each arm therefore gets its own
  `CompiledModelCache` AND its own runner dict.

  A lever that is wired and inert. `extra_calls_delta` per round is the counter, read off
  `ExtraMsaOnDevice` at each round boundary. The OFF rounds are the control: they must move it by
  zero while the ON rounds move it. `--assert-fires` refuses to write a ratio if they do not.
"""
import argparse
import functools
import json
import os
import pathlib
import itertools
import math
import statistics as st
import subprocess
import sys
import time
import traceback

HERE = pathlib.Path(__file__).resolve().parent
_ROOT = HERE.parents[1]
for _p in (str(_ROOT / "perf" / "bcx_round"), str(_ROOT / "perf" / "bcx_predictor"),
           str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import meter as M                                                       # noqa: E402

BC2 = os.environ.get("BCX_BC2", "/home/ttuser/bcx_e2e/bc2")
MONOMER = ("model_1_ptm", "model_2_ptm")

#: Read at TRACE time by the layer_stack dispatcher, set at the round boundary.
ARM = {"on": False}


def git_head():
    return subprocess.run(["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()


def arm_of(r):
    """Round 1 compiles OFF, round 2 compiles ON, then OFF/ON alternate.

    The two compiling rounds pay for a jit and are reported apart from the timed set.
    """
    return r == 2 or (r > 2 and r % 2 == 0)


def per_arm_cache(cls, extra):
    """`cls` with one compiled-gradient cache and one set of AF2 runners per arm.

    Both halves are load-bearing. The cache is BindCraft 2's own memo of the lowered gradient
    program (`bindcraft/af2.py:333`); the runners hold the `jax.jit` underneath it. Sharing
    either one lets the second arm run the first arm's traced program.
    """
    import bindcraft.af2 as bc2_af2

    class PerArm(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._arm_caches = {
                False: (self.gradient_compile_cache, self.alphafold_runners),
                True: (bc2_af2.CompiledModelCache(self.gradient_compile_cache.max_size), {})}

        def _compiled_sequence_gradients(self, *a, **kw):
            self.gradient_compile_cache, self.alphafold_runners = self._arm_caches[ARM["on"]]
            return super()._compiled_sequence_gradients(*a, **kw)
    return PerArm


class ABMeter(M.Meter):
    """`meter.Meter`, plus the arm flip and the counter snapshot at each round boundary.

    The snapshot is also appended to a flushed JSONL as it happens. The full JSON is written at
    the end, and `finally` covers an exception, but neither survives a SIGKILL or an OOM kill --
    and this run costs a card nobody else can have while it holds one. The line on disk is the
    part that survives anything.
    """

    def __init__(self, rounds, extra, live_path=None):
        super().__init__(rounds)
        self.extra = extra
        self.live = open(live_path, "a", buffering=1) if live_path else None

    def on_sequence_gradients_enter(self):
        super().on_sequence_gradients_enter()
        ARM["on"] = arm_of(self.entries)
        ev = {"kind": "arm", "phase": "round", "t0": time.time(),
              "round": self.entries, "extra_msa_on_device": ARM["on"],
              "extra_calls_at_entry": dict(self.extra.calls)}
        M.EVENTS.append(ev)
        if self.live:
            self.live.write(json.dumps({**ev, "utc": time.strftime("%FT%TZ", time.gmtime()),
                                        "load1": os.getloadavg()[0]}) + "\n")
            self.live.flush()
            os.fsync(self.live.fileno())


def permutation_test(off, on):
    """Exact two-sided permutation test on the difference of medians.

    `separated` (every ON round faster than every OFF round) is a range criterion, and a range
    criterion is not calibrated: a real 1.3x with one overlapping pair reads False and understates
    the lever, while at tiny n the same shape of test can reject two identical programs
    (`a-range-containment-criterion-rejects-identical-programs`, where containment at n=2 flagged
    identical arms two thirds of the time).

    So this reports a p-value and, next to it, the smallest p the design could possibly return.
    `min_attainable_p` is the honest resolution: at 3 vs 3 it is 2/20 = 0.1, so "not significant"
    there says nothing about the lever and everything about the rep count.
    """
    n_off, n_on = len(off), len(on)
    if n_off == 0 or n_on == 0:
        return {}
    pool = list(off) + list(on)
    observed = abs(st.median(off) - st.median(on))
    total = hit = 0
    for idx in itertools.combinations(range(len(pool)), n_off):
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(len(pool)) if i not in set(idx)]
        total += 1
        if abs(st.median(a) - st.median(b)) >= observed - 1e-12:
            hit += 1
    return {"perm_p_two_sided": round(hit / total, 5),
            "perm_assignments": total,
            "min_attainable_p": round(2 / total, 5),
            "perm_is_informative": (2 / total) <= 0.05,
            "observed_median_gap_s": round(observed, 3)}


def analyse(events, clock_samples, extra_calls_end):
    starts = [e for e in events if e["kind"] == "round_start"]
    stop = [e["t0"] for e in events if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + stop[:1]
    arm_ev = {e["round"]: e for e in events if e["kind"] == "arm"}
    rows = []
    for i in range(len(bounds) - 1):
        t0, t1 = bounds[i], bounds[i + 1]
        r = starts[i]["round"]
        sg = [e for e in events if e["phase"] == "sequence_gradients" and e.get("round") == r]
        if not sg:
            continue
        s0, s1 = sg[0]["t0"], sg[0]["t1"]

        def dev(prefix):
            return sum(e["dt"] for e in events if e["kind"] == "device"
                       and e["phase"].startswith(prefix) and s0 <= e["t0"] and e["t1"] <= s1)
        evo, extra_s = dev("evo:"), dev("extra:")
        # The counter this round moved: its own entry snapshot against the next round's, or the
        # end-of-run total for the last round.
        at = arm_ev.get(r, {}).get("extra_calls_at_entry", {})
        nxt = arm_ev.get(r + 1, {}).get("extra_calls_at_entry", extra_calls_end)
        delta = {k: nxt.get(k, 0) - at.get(k, 0) for k in ("primal", "taped", "backward")}
        clk = sorted(c for t, c, _ in clock_samples if s0 <= t <= s1)
        load = [ld for t, _, ld in clock_samples if s0 <= t <= s1]
        rows.append({"round": r, "extra_msa_on_device": arm_ev.get(r, {}).get(
            "extra_msa_on_device"),
            "round_wall": round(t1 - t0, 3), "sequence_gradients_s": round(s1 - s0, 3),
            "device_evoformer_s": round(evo, 3), "device_extra_msa_s": round(extra_s, 3),
            "device_share_of_sg": round((evo + extra_s) / (s1 - s0), 4),
            "extra_calls_delta": delta,
            "aiclk_n": len(clk), "aiclk_min": clk[0] if clk else None,
            "aiclk_med": clk[len(clk) // 2] if clk else None,
            "aiclk_max": clk[-1] if clk else None,
            "load1": round(sum(load) / len(load), 1) if load else None})
    timed = [x for x in rows if x["round"] > 2]
    summary = {}
    for on in (False, True):
        xs = [x for x in timed if x["extra_msa_on_device"] is on]
        if not xs:
            continue
        sgs = [x["sequence_gradients_s"] for x in xs]
        walls = [x["round_wall"] for x in xs]
        summary["on" if on else "off"] = {
            "n": len(xs), "sg_median": round(st.median(sgs), 3), "sg_min": min(sgs),
            "sg_max": max(sgs), "sg_spread": round(max(sgs) / min(sgs), 3),
            "round_wall_median": round(st.median(walls), 3), "round_wall_min": min(walls),
            "round_wall_max": max(walls),
            "device_share_median": round(st.median(x["device_share_of_sg"] for x in xs), 4),
            "device_extra_msa_s_median": round(st.median(x["device_extra_msa_s"] for x in xs), 3),
            "extra_backward_calls": sum(x["extra_calls_delta"]["backward"] for x in xs),
            "aiclk_min": min((x["aiclk_min"] for x in xs if x["aiclk_min"]), default=None),
            "aiclk_med_median": st.median([x["aiclk_med"] for x in xs if x["aiclk_med"]] or [0]),
            "load1_median": st.median([x["load1"] for x in xs if x["load1"] is not None] or [0])}
    if "on" in summary and "off" in summary:
        summary["ratio_sg_off_over_on"] = round(
            summary["off"]["sg_median"] / summary["on"]["sg_median"], 3)
        summary["ratio_round_off_over_on"] = round(
            summary["off"]["round_wall_median"] / summary["on"]["round_wall_median"], 3)
        summary["separated"] = summary["on"]["sg_max"] < summary["off"]["sg_min"]
        summary.update(permutation_test(
            [x["sequence_gradients_s"] for x in timed if x["extra_msa_on_device"] is False],
            [x["sequence_gradients_s"] for x in timed if x["extra_msa_on_device"] is True]))
        # The control. An OFF round that runs the card's extra-MSA stack is not a control, and an
        # ON round that does not is not an arm.
        summary["control_clean"] = summary["off"]["extra_backward_calls"] == 0
        summary["arm_fires"] = summary["on"]["extra_backward_calls"] > 0
    return rows, summary


def selftest():
    """`analyse` on synthetic rounds, so the analysis half is not first exercised on a card.

    OFF rounds take 30 s and move no counter, ON rounds take 15 s and move it by one. A correct
    read is 2.0x, arms separated, the control clean, and rounds 1-2 (the compiling pair) out of
    the timed set.
    """
    ev, t, calls = [], 1000.0, {"primal": 0, "taped": 0, "backward": 0}
    for r in range(1, 7):
        on = arm_of(r)
        dur = 15.0 if on else 30.0
        ev.append({"kind": "round_start", "phase": "round", "t0": t, "round": r})
        ev.append({"kind": "arm", "phase": "round", "t0": t, "round": r,
                   "extra_msa_on_device": on, "extra_calls_at_entry": dict(calls)})
        ev.append({"kind": "predictor", "phase": "sequence_gradients", "t0": t, "t1": t + dur,
                   "dt": dur, "round": r})
        ev.append({"kind": "device", "phase": "evo:taped", "t0": t + 1, "t1": t + 6, "dt": 5.0,
                   "round": r})
        if on:
            ev.append({"kind": "device", "phase": "extra:taped", "t0": t + 6, "t1": t + 7,
                       "dt": 1.0, "round": r})
            for k in calls:
                calls[k] += 1
        t += dur
    ev.append({"kind": "round_stop", "phase": "round", "t0": t, "round": 7})
    clk = [(1000.0 + i, 1350, 12.0) for i in range(int(t - 1000))]
    rows, summary = analyse(ev, clk, dict(calls))
    assert [r["round"] for r in rows] == [1, 2, 3, 4, 5, 6], rows
    assert summary["ratio_sg_off_over_on"] == 2.0, summary
    assert summary["separated"] and summary["control_clean"] and summary["arm_fires"], summary
    assert summary["off"]["n"] == summary["on"]["n"] == 2, summary
    assert summary["off"]["extra_backward_calls"] == 0, summary
    assert summary["on"]["extra_backward_calls"] == 2, summary
    assert all(r["extra_calls_delta"]["backward"] == (1 if r["extra_msa_on_device"] else 0)
               for r in rows), rows
    assert summary["on"]["aiclk_med_median"] == 1350, summary
    # 2 vs 2 has C(4,2)=6 assignments, so the smallest p this design can return is 2/6 = 0.333.
    # A separated pair of arms still cannot clear 0.05 here, and the summary must say so rather
    # than let a reader read "separated: true" as significance.
    assert summary["perm_assignments"] == 6, summary
    assert summary["min_attainable_p"] == 0.33333, summary
    assert summary["perm_is_informative"] is False, summary
    assert summary["perm_p_two_sided"] == summary["min_attainable_p"], summary
    # and the test does separate a real gap as far as the design allows
    assert summary["observed_median_gap_s"] == 15.0, summary
    ident = permutation_test([10.0, 10.0, 10.0], [10.0, 10.0, 10.0])
    assert ident["perm_p_two_sided"] == 1.0, ident      # identical arms are never rejected
    assert ident["min_attainable_p"] == 0.1, ident      # 3 vs 3 cannot go below 0.1
    print("selftest ok:", json.dumps({k: summary[k] for k in (
        "ratio_sg_off_over_on", "separated", "control_clean", "arm_fires")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true",
                    help="check `analyse` on synthetic rounds and exit; opens no device")
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--card", default=None, help="pinned card; must be set before ttnn imports")
    ap.add_argument("--assert-fires", action="store_true",
                    help="exit nonzero unless the ON arm moved the counter and the OFF arm "
                         "did not")
    ap.add_argument("--dry-run", action="store_true",
                    help="run everything except the campaign itself and write the artifacts, on "
                         "no card. Exercises the stamp, analyse, the dump paths and the arm "
                         "check, which are otherwise first executed while holding a card")
    ap.add_argument("--out", required=False)
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.out:
        ap.error("--out is required unless --selftest")

    import tt_bio.bindcraft2 as bc2
    import bindcraft.campaign as campaign
    import bindcraft.trajectory as trajectory
    import bindcraft.sequence_optimization as seqopt
    from bindcraft.settings import parse_setting_overrides, read_settings
    from bindcraft.preflight import cleaned_campaign_settings
    from bindcraft.af.alphafold.model import layer_stack as LS

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={project}",
                 # campaign.py:180 compiles the NEXT trajectory's length on a background thread
                 # through sequence_gradients(compile_only=True). It would count as a round, flip
                 # the arm, and take CPU from whichever round it overlaps.
                 "compile_next_length=false"]
    settings = cleaned_campaign_settings(
        read_settings(os.path.join(BC2, "examples", "pdl1.json"),
                      parse_setting_overrides(overrides)))
    campaign.MULTIMER_POOL = MONOMER

    try:
        M.CLOCK = M.Clock(1.0)
        M.CLOCK.start()
    except Exception as exc:
        if not args.dry_run:
            raise
        # No chip, so no AICLK node. A dry run is about the code paths, not the clock.
        print(f"dry-run: no AICLK clock ({type(exc).__name__}: {exc})", flush=True)

        class _NoClock:
            samples, pci, path = [], None, None

            def stop(self):
                pass
        M.CLOCK = _NoClock()
    stamp = {"host": os.uname().nodename, "card": args.card or os.environ.get(
        "TT_VISIBLE_DEVICES"), "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
        "surface": "tt_bio.bindcraft2.predictor(extra_msa=...)",
        "seed": args.seed, "rounds_requested": args.rounds,
        "omp": os.environ.get("OMP_NUM_THREADS"),
        "started_utc": time.strftime("%FT%TZ", time.gmtime()),
        "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    # Resolution before the run, not after: rounds 1-2 compile and the rest alternate, so
    # --rounds 8 leaves 3 vs 3 and a smallest attainable p of 0.1. A run like that can never
    # return a significant result however large the lever is, and trimming rounds to save card
    # time is exactly how someone would reach for it.
    _timed = max(args.rounds - 2, 0)
    _off = _timed // 2
    _c = math.comb(_timed, _off) if _timed else 0
    stamp["min_attainable_p"] = round(2 / _c, 5) if _c else None
    stamp["design_is_informative"] = bool(_c) and (2 / _c) <= 0.05
    if not stamp["design_is_informative"]:
        print(f"WARNING: --rounds {args.rounds} leaves {_off} vs {_timed - _off} timed rounds, "
              f"smallest attainable p {stamp['min_attainable_p']}. This design cannot return a "
              f"significant result. Use --rounds 10 or more.", flush=True)
    print(json.dumps(stamp, indent=1), flush=True)

    # BindCraft 2's own layer_stack, captured BEFORE the predictor patches it. The OFF arm's
    # extra-MSA stack goes through this one.
    real = LS.layer_stack

    with bc2.predictor(trunk="device", card=args.card, checkpoints=args.params,
                       extra_msa=True) as build:
        evo, extra = build.evoformer, build.extra_msa
        assert extra is not None, "predictor(extra_msa=True) built no ExtraMsaOnDevice"
        spliced = LS.layer_stack

        mt = ABMeter(args.rounds, extra, live_path=os.path.join(project, "rounds.jsonl"))
        M.install(mt, bc2, bc2.design_model_class(), trajectory, seqopt)
        # meter.install tags Evoformer calls with bare names; retag them, and tag the extra
        # stack, so a round's device seconds split by which stack spent them.
        for cls, tag in ((bc2.EvoformerOnDevice, "evo"), (bc2.ExtraMsaOnDevice, "extra")):
            for name in ("_primal", "_taped", "_backward"):
                orig = getattr(cls, name)

                def make(orig, label):
                    def wrapper(self, *a, **kw):
                        t0 = time.time()
                        n0 = len(M.EVENTS)
                        try:
                            return orig(self, *a, **kw)
                        finally:
                            # Drop the untagged event meter.install's own wrapper just wrote.
                            del M.EVENTS[n0:]
                            M.ev("device", label, t0, time.time(), round=mt.entries)
                    return wrapper
                setattr(cls, name, make(orig, f"{tag}:{name.lstrip('_')}"))

        campaign.AlphaFoldDesignModel = functools.partial(
            per_arm_cache(bc2.design_model_class(), extra),
            trunk="device", pool=build.pool)

        # The arm is read at TRACE time: an extra-MSA stack traced while ARM is ON takes the
        # card's, one traced while it is OFF takes BindCraft 2's. Everything else, the Evoformer
        # included, always takes the spliced factory, so the swap under test is the only thing
        # that moves between arms.
        traced = {"on": 0, "off": 0}

        def by_arm(num_layers, *a, **kw):
            on = ARM["on"]
            made = (spliced if on else real)(num_layers, *a, **kw)
            rest = spliced(num_layers, *a, **kw)

            def choose(fn):
                if getattr(fn, "__name__", None) == "extra_msa_stack_fn":
                    traced["on" if on else "off"] += 1
                    return made(fn)
                return rest(fn)
            return choose
        LS.layer_stack = by_arm
        t0 = time.time()
        stopped, failure = None, None
        try:
            if args.dry_run:
                raise M.StopAfterRounds("dry run: the campaign was not started")
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(BC2, "bindcraft", "weights",
                                                            "proteinmpnn", "weights_neutral"),
                                  max_trajectories=1)
        except M.StopAfterRounds as stop:
            stopped = str(stop)
        except BaseException as exc:
            # Nine rounds that completed are worth more than a clean traceback. Bank them, then
            # re-raise below so the failure is still a failure.
            failure = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()
        finally:
            LS.layer_stack = spliced
            M.CLOCK.stop()

    rows, summary = analyse(M.EVENTS, M.CLOCK.samples, dict(extra.calls))
    stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                  "failure": failure, "rounds_completed": len(rows),
                  "evo_calls": dict(evo.calls), "extra_calls": dict(extra.calls),
                  "extra_swapped": list(extra.swapped), "extra_traces": traced,
                  "extra_mask_seen": dict(extra.mask_seen),
                  "loadavg_end": os.getloadavg(),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
    M.dump(os.path.join(project, "round_events.json"), stamp)
    out = {"stamp": stamp, "summary": summary, "rounds": rows}
    pathlib.Path(project, "round_ab.json").write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        print(json.dumps(r), flush=True)
    print(json.dumps(summary, indent=1), flush=True)

    if failure is not None:
        raise SystemExit(f"the campaign raised after {len(rows)} rounds; results for those "
                         f"rounds are banked in {project}. {failure}")

    if args.dry_run:
        print(f"dry-run OK: wrote {project}/round_ab.json and round_events.json, "
              f"{len(rows)} rounds (0 expected), arm check skipped", flush=True)
        return

    if args.assert_fires:
        bad = []
        if not stamp.get("design_is_informative"):
            bad.append(f"the design cannot return a significant result: smallest attainable p is "
                       f"{stamp.get('min_attainable_p')}, needs --rounds 10 or more")
        if not summary.get("arm_fires"):
            bad.append(f"the ON arm never reached the card: traces {traced}, "
                       f"calls {dict(extra.calls)}")
        if not summary.get("control_clean"):
            bad.append("the OFF arm ran the card's extra-MSA stack, so it is not a control")
        if bad:
            raise SystemExit("ARM CHECK FAILED: " + "; ".join(bad))


if __name__ == "__main__":
    main()
