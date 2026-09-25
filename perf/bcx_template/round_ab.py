#!/usr/bin/env python3
"""BindCraft 2 gradient rounds with the template pair stack OFF and ON, interleaved.

The arm is `perf/bcx_extramsa/round_ab.py`'s with one variable moved: the extra-MSA stack is on
the card in BOTH arms, because that swap is `bcx-extramsa`'s GO at 2.156x and the program this
row is aimed at is the one the campaign is keeping. What changes between rounds is whether
`modules.py:247`'s template `layer_stack` is BindCraft 2's JAX or `splice.
TemplatePairStackOnDevice`.

Two traps this inherits, both already paid for by another row:

* `RunModel.apply` is a `jax.jit` built once per runner (`af/alphafold/model/model.py:96`) and
  keyed by input shapes, so a second outer program reuses the first's trace and BOTH arms run
  the first arm's code. Each arm gets its own runner dict and its own `CompiledModelCache`, and
  the stamp carries the per-arm trace count and the card's call count so the artifact proves the
  arms differed.
* The arm is read at TRACE time, not call time: a template stack traced while ARM is ON takes
  the device stack, one traced while it is OFF takes BindCraft 2's.

`compile_next_length=false` because `campaign.py:180` compiles the next trajectory's length in a
background thread through `sequence_gradients(compile_only=True)`, which would count as a round,
flip the arm, and take CPU from whichever round it overlaps.
"""
import argparse
import functools
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
import bindcraft.af2 as bc2_af2                                        # noqa: E402
import bindcraft.campaign as campaign                                  # noqa: E402
import bindcraft.trajectory as trajectory                              # noqa: E402
import bindcraft.sequence_optimization as seqopt                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from run_round import MONOMER, git_head                                # noqa: E402

ARM = {"on": False}
TEMPLATE_BLOCKS = 2


def arm_of(r):
    """Round 1 compiles OFF, round 2 compiles ON, then OFF/ON alternate."""
    return r == 2 or (r > 2 and r % 2 == 0)


class ABMeter(M.Meter):
    def on_sequence_gradients_enter(self):
        super().on_sequence_gradients_enter()
        ARM["on"] = arm_of(self.entries)
        M.EVENTS.append({"kind": "arm", "phase": "round", "t0": time.time(),
                         "round": self.entries, "template_on_device": ARM["on"]})


def per_arm_cache(cls):
    """`cls` with one compiled-gradient cache and one set of AF2 runners per arm."""
    class PerArm(cls):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._arm_caches = {False: (self.gradient_compile_cache, self.alphafold_runners),
                                True: (bc2_af2.CompiledModelCache(self.gradient_compile_cache
                                                                  .max_size), {})}

        def _compiled_sequence_gradients(self, *a, **kw):
            self.gradient_compile_cache, self.alphafold_runners = self._arm_caches[ARM["on"]]
            return super()._compiled_sequence_gradients(*a, **kw)
    return PerArm


def analyse(events, clock_samples):
    starts = [e for e in events if e["kind"] == "round_start"]
    stop = [e["t0"] for e in events if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + stop[:1]
    arms = {e["round"]: e["template_on_device"] for e in events if e["kind"] == "arm"}
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
        evo, extra, tmpl = dev("evo:"), dev("extra:"), dev("tmpl:")
        clk = sorted(c for t, c, _ in clock_samples if s0 <= t <= s1)
        load = [ld for t, _, ld in clock_samples if s0 <= t <= s1]
        rows.append({"round": r, "template_on_device": arms.get(r),
                     "round_wall": round(t1 - t0, 3),
                     "sequence_gradients_s": round(s1 - s0, 3),
                     "device_evoformer_s": round(evo, 3), "device_extra_msa_s": round(extra, 3),
                     "device_template_s": round(tmpl, 3),
                     "device_share_of_sg": round((evo + extra + tmpl) / (s1 - s0), 4),
                     "aiclk_n": len(clk), "aiclk_min": clk[0] if clk else None,
                     "aiclk_med": clk[len(clk) // 2] if clk else None,
                     "aiclk_max": clk[-1] if clk else None,
                     "load1": round(sum(load) / len(load), 1) if load else None})
    timed = [x for x in rows if x["round"] > 2]
    summary = {}
    for on in (False, True):
        xs = [x for x in timed if x["template_on_device"] is on]
        if not xs:
            continue
        sgs = sorted(x["sequence_gradients_s"] for x in xs)
        walls = [x["round_wall"] for x in xs]
        summary["on" if on else "off"] = {
            "n": len(xs), "sg_median": round(st.median(sgs), 3), "sg_min": sgs[0],
            "sg_max": sgs[-1], "sg_all": sgs,
            "sg_spread": round(sgs[-1] / sgs[0], 3),
            "round_wall_median": round(st.median(walls), 3), "round_wall_min": min(walls),
            "round_wall_max": max(walls),
            "device_template_median": round(st.median(x["device_template_s"] for x in xs), 3),
            "device_share_median": round(st.median(x["device_share_of_sg"] for x in xs), 4),
            "aiclk_min": min((x["aiclk_min"] for x in xs if x["aiclk_min"]), default=None),
            "aiclk_med_median": st.median([x["aiclk_med"] for x in xs if x["aiclk_med"]] or [0]),
            "load1_median": st.median([x["load1"] for x in xs if x["load1"] is not None] or [0])}
    # A 1.1x effect against a 1.2x round-to-round spread does not separate on six rounds an
    # arm, so the arms are also read as matched pairs: each ON round against the mean of the
    # OFF rounds either side of it, which cancels any drift slower than one round. The rank sum
    # is the unpaired reading beside it; both are reported, neither is chosen afterwards.
    by_round = {x["round"]: x for x in timed}
    pairs = []
    for x in timed:
        if not x["template_on_device"]:
            continue
        nb = [by_round[r]["sequence_gradients_s"] for r in (x["round"] - 1, x["round"] + 1)
              if r in by_round and by_round[r]["template_on_device"] is False]
        if nb:
            pairs.append(round(sum(nb) / len(nb) - x["sequence_gradients_s"], 3))
    if pairs:
        summary["paired"] = {"n": len(pairs), "diffs_off_minus_on": sorted(pairs),
                             "median_s": round(st.median(pairs), 3),
                             "faster_with_template_on_card": sum(1 for d in pairs if d > 0)}
    ranked = sorted(timed, key=lambda x: x["sequence_gradients_s"])
    n_on = sum(1 for x in timed if x["template_on_device"])
    n_off = len(timed) - n_on
    rank_on = sum(i + 1 for i, x in enumerate(ranked) if x["template_on_device"])
    if n_on and n_off:
        u_on = rank_on - n_on * (n_on + 1) / 2
        summary["rank"] = {"n_on": n_on, "n_off": n_off, "rank_sum_on": rank_on,
                           "u_on": u_on, "u_max": n_on * n_off,
                           "fraction_of_pairs_on_is_faster": round(1 - u_on / (n_on * n_off), 3)}
    if "on" in summary and "off" in summary:
        summary["ratio_sg_off_over_on"] = round(summary["off"]["sg_median"]
                                                / summary["on"]["sg_median"], 3)
        summary["ratio_round_off_over_on"] = round(summary["off"]["round_wall_median"]
                                                   / summary["on"]["round_wall_median"], 3)
        summary["seconds_saved_median"] = round(summary["off"]["sg_median"]
                                                - summary["on"]["sg_median"], 3)
        summary["separated"] = summary["on"]["sg_max"] < summary["off"]["sg_min"]
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
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

    import ttbio_predictor as T
    campaign.AlphaFoldDesignModel = functools.partial(
        per_arm_cache(T.TTBioAlphaFoldDesignModel), trunk="device")

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
             "seed": args.seed, "rounds_requested": args.rounds,
             "omp": os.environ.get("OMP_NUM_THREADS"),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    import afgrad as _A
    import splice
    from splice import (EvoformerOnDevice, ExtraMsaOnDevice, TemplatePairStackOnDevice,
                        evoformer_on_device)
    mt = ABMeter(args.rounds)
    M.install(mt, splice, T.TTBioAlphaFoldDesignModel, trajectory, seqopt)
    # meter.install tags each stack's calls with its own prefix already; retag them onto the
    # round so a device second lands in the round that spent it.
    for cls, tag, names in ((EvoformerOnDevice, "evo", ("_primal", "_taped", "_backward")),
                            (ExtraMsaOnDevice, "extra", ("_primal", "_taped", "_backward")),
                            (TemplatePairStackOnDevice, "tmpl", ("_forward",))):
        for name in names:
            orig = getattr(cls, name)

            def make(orig, label):
                def wrapper(self, *a, **kw):
                    t0 = time.time()
                    n0 = len(M.EVENTS)
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        del M.EVENTS[n0:]      # drop meter.install's own untagged event
                        M.ev("device", label, t0, time.time(), round=mt.entries)
                return wrapper
            setattr(cls, name, make(orig, f"{tag}:{name.lstrip('_')}"))

    _dm, _ = _A.load_models(_A.DEFAULT_PARAMS, template=True)
    _dev = _A.Dev(_dm.to_device())
    evo = EvoformerOnDevice(_dev, k_evo=48)
    extra = ExtraMsaOnDevice(_dev, k_extra=4)
    tmpl = TemplatePairStackOnDevice(_dev, k_template=TEMPLATE_BLOCKS)

    # Two factories built from the same splice, differing in one argument. Each context manager
    # is entered and left before the next one is entered, so both closed over the REAL
    # `layer_stack` and either can be dispatched to per round.
    from bindcraft.af.alphafold.model import layer_stack as LS
    real = LS.layer_stack
    made = {}
    for key, kw in (("on", {"template": tmpl}), ("off", {})):
        ctx = evoformer_on_device(evo, extra_msa=extra, **kw)
        ctx.__enter__()
        made[key] = LS.layer_stack
        ctx.__exit__(None, None, None)
    assert LS.layer_stack is real

    traced = {"on": 0, "off": 0}

    def by_arm(num_layers, *a, **kw):
        arm = "on" if ARM["on"] else "off"
        built = made[arm](num_layers, *a, **kw)

        def choose(fn):
            if (getattr(fn, "__name__", None) == "block"
                    and int(num_layers) == TEMPLATE_BLOCKS):
                traced[arm] += 1
            return built(fn)
        return choose

    LS.layer_stack = by_arm
    t0 = time.time()
    stopped = None
    try:
        campaign.run_campaign(settings, project, af2_weights=args.params,
                              mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                        "proteinmpnn", "weights_neutral"),
                              max_trajectories=1)
    except M.StopAfterRounds as stop:
        stopped = str(stop)
    finally:
        LS.layer_stack = real
        M.CLOCK.stop()

    rows, summary = analyse(M.EVENTS, M.CLOCK.samples)
    on_rounds = sum(1 for r in rows if r["template_on_device"])
    if on_rounds and (traced["on"] == 0 or tmpl.calls["forward"] < on_rounds):
        raise RuntimeError(f"{on_rounds} ON rounds but the device template stack traced "
                           f"{traced['on']} times and ran {tmpl.calls}: the ON arm is JAX")
    stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                  "evo_calls": dict(evo.calls), "extra_calls": dict(extra.calls),
                  "template_calls": dict(tmpl.calls), "template_traces": traced,
                  "template_swapped": tmpl.swapped, "template_channels": tmpl.channels,
                  "template_shapes": sorted(map(str, tmpl.shapes)),
                  "template_dropout_seen": sorted(tmpl.dropout_seen),
                  "loadavg_end": os.getloadavg(),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
    M.dump(os.path.join(project, "round_events.json"), stamp)
    out = {"stamp": stamp, "summary": summary, "rounds": rows}
    pathlib.Path(project, "round_ab.json").write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        print(json.dumps(r), flush=True)
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
