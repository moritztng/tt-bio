#!/usr/bin/env python3
"""BindCraft 2 gradient rounds with the extra-MSA swap OFF and ON, interleaved in one process.

The arm is `perf/bcx_seam/run_seam.py`'s: BindCraft 2's `campaign.py` drives, the Evoformer
runs on the card for every call, `perf/bcx_round/meter.py` timestamps the seams, one card, one
process, seed 100. What changes between rounds is one thing, whether `modules.py:1528`'s
extra-MSA `layer_stack` is BindCraft 2's JAX or `splice.ExtraMsaOnDevice`.

BindCraft 2 caches its compiled gradient program on the model instance
(`bindcraft/af2.py:333`), and the swap acts at trace time, so the two arms need two programs.
Each arm gets its own `CompiledModelCache`; the arm is chosen at the round boundary, before the
round's `lower()`, and the factory reads it while tracing. Rounds 1 and 2 compile the OFF and ON
programs and are reported apart; from round 3 the arms alternate OFF, ON, OFF, ON.

Per round: the wall between consecutive `sequence_gradients` entries (bcx-round's round), the
`sequence_gradients` call itself, device seconds inside it split Evoformer / extra-MSA, AICLK
sampled every second during it, and the 1-minute loadavg.
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


def arm_of(r):
    """Round 1 compiles OFF, round 2 compiles ON, then OFF/ON alternate."""
    return r == 2 or (r > 2 and r % 2 == 0)


class ABMeter(M.Meter):
    def on_sequence_gradients_enter(self):
        super().on_sequence_gradients_enter()
        ARM["on"] = arm_of(self.entries)
        M.EVENTS.append({"kind": "arm", "phase": "round", "t0": time.time(),
                         "round": self.entries, "extra_msa_on_device": ARM["on"]})


def per_arm_cache(cls):
    """`cls` with one compiled-gradient cache and one set of AF2 runners per arm.

    The runners matter as much as the cache. `RunModel.apply` is a `jax.jit` built once per
    runner (`af/alphafold/model/model.py:96`), and JAX keeps its traced jaxpr keyed by input
    shapes, so a second outer program that calls the same runner reuses the first one's trace
    and the factory is never asked again. The first version of this script shared the runners:
    the extra-MSA stack traced once, in the OFF arm, and every ON round ran BindCraft 2's JAX.
    """
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
    arms = {e["round"]: e["extra_msa_on_device"] for e in events if e["kind"] == "arm"}
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
        evo = dev("evo:")
        extra = dev("extra:")
        clk = sorted(c for t, c, _ in clock_samples if s0 <= t <= s1)
        load = [ld for t, _, ld in clock_samples if s0 <= t <= s1]
        rows.append({"round": r, "extra_msa_on_device": arms.get(r), "round_wall": round(t1 - t0, 3),
                     "sequence_gradients_s": round(s1 - s0, 3), "device_evoformer_s": round(evo, 3),
                     "device_extra_msa_s": round(extra, 3),
                     "device_share_of_sg": round((evo + extra) / (s1 - s0), 4),
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
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    project = args.out
    pathlib.Path(project).mkdir(parents=True, exist_ok=True)
    overrides = [f"campaign_seed={args.seed}", "max_trajectories=1",
                 "validation_model=monomer", 'design_models=["model_1_ptm"]',
                 'validation_models=["model_2_ptm"]', f"project_folder={project}",
                 # campaign.py:180 compiles the NEXT trajectory's length in a background thread
                 # through sequence_gradients(compile_only=True). It would count as a round, flip
                 # the arm, and take CPU from whichever round it overlaps. One trajectory is run.
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
    from splice import EvoformerOnDevice, ExtraMsaOnDevice, evoformer_on_device
    mt = ABMeter(args.rounds)
    M.install(mt, splice, T.TTBioAlphaFoldDesignModel, trajectory, seqopt)
    # meter.install tags Evoformer calls with bare names; retag them, and tag the extra stack.
    for cls, tag in ((EvoformerOnDevice, "evo"), (ExtraMsaOnDevice, "extra")):
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

    _dm, _ = _A.load_models(_A.DEFAULT_PARAMS)
    _dev = _A.Dev(_dm.to_device())
    evo = EvoformerOnDevice(_dev, k_evo=48)
    extra = ExtraMsaOnDevice(_dev, k_extra=4)

    # The arm is read at TRACE time: inside the splice, an extra-MSA stack traced while ARM is
    # ON takes the splice's device stack and one traced while it is OFF takes BindCraft 2's.
    from bindcraft.af.alphafold.model import layer_stack as LS
    real = LS.layer_stack
    traced = {"on": 0, "off": 0}
    with evoformer_on_device(evo, extra_msa=extra):
        spliced = LS.layer_stack

        def by_arm(num_layers, *a, **kw):
            on = ARM["on"]
            made = (spliced if on else real)(num_layers, *a, **kw)
            evo_or_rest = spliced(num_layers, *a, **kw)

            def choose(fn):
                if getattr(fn, "__name__", None) == "extra_msa_stack_fn":
                    traced["on" if on else "off"] += 1
                    return made(fn)
                return evo_or_rest(fn)
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
            LS.layer_stack = spliced
            M.CLOCK.stop()
    rows, summary = analyse(M.EVENTS, M.CLOCK.samples)
    on_rounds = sum(1 for r in rows if r["extra_msa_on_device"])
    if on_rounds and (traced["on"] == 0 or extra.calls["backward"] < on_rounds):
        raise RuntimeError(f"{on_rounds} ON rounds but the device extra-MSA stack traced "
                           f"{traced['on']} times and ran {extra.calls}: the ON arm is JAX")
    stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                  "evo_calls": dict(evo.calls), "extra_calls": dict(extra.calls),
                  "extra_traces": traced,
                  "extra_mask_seen": {**extra.mask_seen,
                                      "shapes": sorted(map(list, extra.mask_seen["shapes"]))},
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
