#!/usr/bin/env python3
"""BindCraft 2 gradient rounds with the template pair stack OFF and ON, interleaved.

The arm is `perf/bcx_extramsa/round_ab.py`'s with the extra-MSA swap held ON in every round,
because that is the round this row has to improve: 16.87 s median, not 36.37. What changes
between rounds is one thing, whether `modules.py:247`'s template `layer_stack` runs BindCraft
2's JAX or `splice.TemplatePairStackOnDevice`.

BindCraft 2 caches its compiled gradient program on the model instance and the swap acts at
trace time, so the two arms need two programs AND two sets of AF2 runners -- `RunModel.apply`
is a `jax.jit` built once per runner, so a shared runner has the second arm reuse the first
arm's trace and never ask the factory again. That bug read 1.055x on a 2.16x lever once
already (`state/bcx-extramsa.md`), so the count is asserted at the end rather than assumed.
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

import torch                                                           # noqa: E402
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


def arm_of(r):
    """Round 1 compiles OFF, round 2 compiles ON, then OFF/ON alternate."""
    return r == 2 or (r > 2 and r % 2 == 0)


class ABMeter(M.Meter):
    def __init__(self, rounds, profile=(), trace_dir=None):
        super().__init__(rounds)
        self.profile, self.trace_dir, self.tracing = profile, trace_dir, False

    def on_sequence_gradients_enter(self):
        import jax
        r = self.entries + 1
        if self.profile and self.tracing and r > self.profile[1]:
            jax.profiler.stop_trace()
            self.tracing = False
            M.EVENTS.append({"kind": "trace_stop", "phase": "round", "t0": time.time(),
                             "round": r})
        super().on_sequence_gradients_enter()
        ARM["on"] = arm_of(self.entries)
        M.EVENTS.append({"kind": "arm", "phase": "round", "t0": time.time(),
                         "round": self.entries, "template_on_device": ARM["on"]})
        if self.profile and r == self.profile[0]:
            jax.profiler.start_trace(self.trace_dir)
            self.tracing = True
            M.EVENTS.append({"kind": "trace_start", "phase": "round", "t0": time.time(),
                             "round": r})


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


def device_model():
    """tt-bio's AF2 with the template pair stack built, which `afgrad.load_models` skips."""
    import afgrad as _A
    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    state = load_af2_state_dict(_A.DEFAULT_PARAMS)
    return load_af2_device_model(state, template=True, trunk_dtype=torch.bfloat16)


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
        sgs = [x["sequence_gradients_s"] for x in xs]
        walls = [x["round_wall"] for x in xs]
        summary["on" if on else "off"] = {
            "n": len(xs), "sg_median": round(st.median(sgs), 3), "sg_min": min(sgs),
            "sg_max": max(sgs), "sg_spread": round(max(sgs) / min(sgs), 3),
            "sg_all": sorted(sgs),
            "round_wall_median": round(st.median(walls), 3), "round_wall_min": min(walls),
            "round_wall_max": max(walls),
            "device_template_median": round(st.median(x["device_template_s"] for x in xs), 3),
            "device_share_median": round(st.median(x["device_share_of_sg"] for x in xs), 4),
            "aiclk_min": min((x["aiclk_min"] for x in xs if x["aiclk_min"]), default=None),
            "aiclk_med_median": st.median([x["aiclk_med"] for x in xs if x["aiclk_med"]] or [0]),
            "load1_median": st.median([x["load1"] for x in xs if x["load1"] is not None] or [0])}
    if "on" in summary and "off" in summary:
        summary["ratio_sg_off_over_on"] = round(summary["off"]["sg_median"]
                                                / summary["on"]["sg_median"], 3)
        summary["ratio_round_off_over_on"] = round(summary["off"]["round_wall_median"]
                                                   / summary["on"]["round_wall_median"], 3)
        summary["separated"] = summary["on"]["sg_max"] < summary["off"]["sg_min"]
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--profile", default="", help="A,B: jax.profiler over rounds A..B")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--no-extra-msa", action="store_true",
                    help="leave the extra-MSA stack in JAX; the default holds it on card")
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
    profile = tuple(int(x) for x in args.profile.split(",")) if args.profile else ()
    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
             "seed": args.seed, "rounds_requested": args.rounds, "profile": profile,
             "extra_msa_on_device": not args.no_extra_msa,
             "omp": os.environ.get("OMP_NUM_THREADS"),
             "xla_flags": os.environ.get("XLA_FLAGS"),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    import afgrad as _A
    import splice
    from splice import (EvoformerOnDevice, ExtraMsaOnDevice, TemplatePairStackOnDevice,
                        evoformer_on_device, template_pair_stack_on_device)
    mt = ABMeter(args.rounds, profile, os.path.join(project, "trace"))
    M.install(mt, splice, T.TTBioAlphaFoldDesignModel, trajectory, seqopt)
    for cls, tag in ((EvoformerOnDevice, "evo"), (ExtraMsaOnDevice, "extra"),
                     (TemplatePairStackOnDevice, "tmpl")):
        for name in ("_primal", "_taped", "_backward"):
            orig = getattr(cls, name, None)
            if orig is None:
                continue

            def make(orig, label):
                def wrapper(self, *a, **kw):
                    t0 = time.time()
                    n0 = len(M.EVENTS)
                    try:
                        return orig(self, *a, **kw)
                    finally:
                        del M.EVENTS[n0:]
                        M.ev("device", label, t0, time.time(), round=mt.entries)
                return wrapper
            setattr(cls, name, make(orig, f"{tag}:{name.lstrip('_')}"))

    _dev = _A.Dev(device_model())
    evo = EvoformerOnDevice(_dev, k_evo=48)
    extra = None if args.no_extra_msa else ExtraMsaOnDevice(_dev, k_extra=4)
    tmpl = TemplatePairStackOnDevice(_dev, k_blocks=2)

    from bindcraft.af.alphafold.model import modules
    real_tps = modules.TemplatePairStack.__call__
    t0 = time.time()
    stopped = None
    traced = {"on": 0, "off": 0}
    with evoformer_on_device(evo, extra_msa=extra):
        with template_pair_stack_on_device(tmpl):
            spliced = modules.TemplatePairStack.__call__

            def by_arm(self, pair_act, pair_mask, use_dropout, safe_key=None):
                on = ARM["on"]
                traced["on" if on else "off"] += 1
                fn = spliced if on else real_tps
                return fn(self, pair_act, pair_mask, use_dropout, safe_key)
            modules.TemplatePairStack.__call__ = by_arm
            try:
                campaign.run_campaign(settings, project, af2_weights=args.params,
                                      mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                                "proteinmpnn",
                                                                "weights_neutral"),
                                      max_trajectories=1)
            except M.StopAfterRounds as stop:
                stopped = str(stop)
            finally:
                modules.TemplatePairStack.__call__ = spliced
                M.CLOCK.stop()
                if mt.tracing:
                    import jax
                    jax.profiler.stop_trace()
    rows, summary = analyse(M.EVENTS, M.CLOCK.samples)
    on_rounds = sum(1 for r in rows if r["template_on_device"])
    if on_rounds and (traced["on"] == 0 or tmpl.calls["primal"] < on_rounds):
        raise RuntimeError(f"{on_rounds} ON rounds but the device template stack traced "
                           f"{traced['on']} times and ran {tmpl.calls}: the ON arm is JAX")
    stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                  "evo_calls": dict(evo.calls),
                  "extra_calls": dict(extra.calls) if extra else None,
                  "tmpl_calls": dict(tmpl.calls), "tmpl_traces": traced,
                  "tmpl_swapped": tmpl.swapped, "tmpl_dropout": dict(tmpl.dropout_seen),
                  "loadavg_end": os.getloadavg(),
                  "finished_utc": time.strftime("%FT%TZ", time.gmtime())})
    M.dump(os.path.join(project, "round_events.json"), stamp)
    out = {"stamp": stamp, "summary": summary, "rounds": rows}
    pathlib.Path(project, "round_ab.json").write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        print(json.dumps(r), flush=True)
    print(json.dumps(summary, indent=1), flush=True)
    print(json.dumps({k: stamp[k] for k in ("tmpl_calls", "tmpl_traces", "tmpl_swapped",
                                            "tmpl_dropout")}, indent=1), flush=True)


if __name__ == "__main__":
    main()
