#!/usr/bin/env python3
"""bcx-bwbytes: real BindCraft 2 gradient rounds with the two byte levers OFF and ON.

The arm is `perf/bcx_tmplseam/round_ab.py`'s minus the template splice, because that row measured
no separation from it: the round this one has to improve is the extra-MSA swap held ON and the
template pair stack left in JAX, 11.29 s median at n=211.

Unlike every earlier A/B on this campaign the two arms share ONE compiled gradient program and
ONE set of AF2 runners, and that is not a shortcut. The levers live inside our own ttnn backward
closures, below anything `jax.jit` traces, so nothing about the JAX side differs between rounds
and a per-arm cache would only add a compile. What has to be asserted instead is that the levers
FIRE: every round counts the reblock kernels served, the leading sums served, and the ones each
gate declined, and the run refuses to report if an ON round served zero.

This harness also carries the row's CEILING. `device_share_of_sg` is the device seconds inside
`sequence_gradients` over the wall of the same window, per round, on the tree it actually ran.

**The primary metric here is `device_evoformer_s`, not the round wall, and that is a result rather
than a convenience.** `perf/bcx_bwbytes/power.py` reads bcx-tmplseam's twelve rounds on this same
tree: over ten timed rounds the round wall has CV 5.80 % while the card's own Evoformer seconds
have CV 3.27 %, and the lever reaches 9.1 % of the device time against 7.1 % of the round. At
80 % power and 5 % two-sided that is **13 reps per arm on the wall and 3 on the device metric**.
A twelve-round A/B -- five per arm, which is what every earlier row on this campaign ran -- cannot
resolve the predicted 1.071x on the wall, and would report "no separation" for a lever that is
working. bcx-tmplseam demonstrated exactly that empirically: p = 0.40 on a 0.4 s effect. So
`--rounds` defaults to 28, both metrics are reported, and the device one leads.
"""
import argparse
import collections
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
import bindcraft.campaign as campaign                                  # noqa: E402
import bindcraft.trajectory as trajectory                              # noqa: E402
import bindcraft.sequence_optimization as seqopt                       # noqa: E402
from bindcraft.settings import parse_setting_overrides, read_settings   # noqa: E402
from bindcraft.preflight import cleaned_campaign_settings              # noqa: E402
from run_round import MONOMER, git_head                                # noqa: E402

ARM = {"on": False}
COUNT = collections.Counter()


def arm_of(r):
    """Round 1 and 2 are the warm-up either way; then OFF/ON alternate from round 3."""
    return r > 2 and r % 2 == 0


def install_levers(rows_on, precision):
    """Wire both levers to `ARM` and count every decision they take.

    A count that only ever goes up on one arm proves nothing -- the control has to MOVE it -- so
    both the served and the declined side are counted, and the summary reports both arms.
    """
    from tt_bio import autograd as ag, reblock_permute as R, taped_ttnn as T

    real_back, real_fwd = R.reblock_permute_back, R.reblock_permute
    real_tree, real_use = ag._pairwise_sum0, ag._use_tree

    def back(x, mc=None, device=None):
        COUNT[f"reblock_back:{'on' if ARM['on'] else 'off'}"] += 1
        return real_back(x, mc, device)

    def fwd(x, mc=None, device=None):
        COUNT[f"reblock_fwd:{'on' if ARM['on'] else 'off'}"] += 1
        return real_fwd(x, mc, device)

    def tree(t):
        COUNT[f"tree:{'on' if ARM['on'] else 'off'}"] += 1
        return real_tree(t)

    def use(t):
        ok = real_use(t)
        COUNT[f"sum0_{'tree' if ok else 'ttnn'}:{'on' if ARM['on'] else 'off'}"] += 1
        COUNT[f"sum0_rows:{int(t.shape[0])}"] += 1
        return ok

    R.reblock_permute_back, R.reblock_permute = back, fwd
    ag._pairwise_sum0, ag._use_tree = tree, use

    real_dx = ag.softmax_bw_dx

    def dx(y, g, dim=-1, config=None):
        COUNT[f"softmax_bw:{'on' if ARM['on'] else 'off'}:"
              f"{str(y.dtype).split('.')[-1]}"] += 1
        return real_dx(y, g, dim=dim, config=config)

    ag.softmax_bw_dx = dx

    def apply():
        T.PERMUTE_BW_REBLOCK = ARM["on"]
        ag.LEADING_SUM_TREE_ROWS = rows_on if ARM["on"] else 1 << 30
        ag.SOFTMAX_BW_DTYPE = "bf16" if (ARM["on"] and precision) else "keep"
        ag.FANIN_WIDEN_INCOMING = not (ARM["on"] and precision)
    return apply


class ABMeter(M.Meter):
    def __init__(self, rounds, apply):
        super().__init__(rounds)
        self.apply = apply

    def on_sequence_gradients_enter(self):
        super().on_sequence_gradients_enter()
        ARM["on"] = arm_of(self.entries)
        self.apply()
        M.EVENTS.append({"kind": "arm", "phase": "round", "t0": time.time(),
                         "round": self.entries, "levers_on": ARM["on"]})


def device_model():
    import afgrad as _A
    from tt_bio.af2 import load_af2_device_model
    from tt_bio.af2_weights import load_af2_state_dict
    return load_af2_device_model(load_af2_state_dict(_A.DEFAULT_PARAMS), template=False,
                                 trunk_dtype=torch.bfloat16)


def analyse(events, clock_samples):
    starts = [e for e in events if e["kind"] == "round_start"]
    stop = [e["t0"] for e in events if e["kind"] == "round_stop"]
    bounds = [e["t0"] for e in starts] + stop[:1]
    arms = {e["round"]: e["levers_on"] for e in events if e["kind"] == "arm"}
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
        evo, extra = dev("evo:"), dev("extra:")
        bwd = sum(e["dt"] for e in events if e["kind"] == "device"
                  and e["phase"].endswith(":backward") and s0 <= e["t0"] and e["t1"] <= s1)
        clk = sorted(c for t, c in clock_samples if s0 <= t <= s1)
        rows.append({"round": r, "levers_on": arms.get(r), "round_wall": round(t1 - t0, 3),
                     "sequence_gradients_s": round(s1 - s0, 3),
                     "device_evoformer_s": round(evo, 3), "device_extra_msa_s": round(extra, 3),
                     "device_backward_s": round(bwd, 3),
                     "device_share_of_sg": round((evo + extra) / (s1 - s0), 4),
                     "aiclk_n": len(clk), "aiclk_min": clk[0] if clk else None,
                     "aiclk_med": clk[len(clk) // 2] if clk else None,
                     "aiclk_max": clk[-1] if clk else None,
                     "load1": round(os.getloadavg()[0], 1)})
    timed = [x for x in rows if x["round"] > 2]
    summary = {}
    for on in (False, True):
        xs = [x for x in timed if x["levers_on"] is on]
        if not xs:
            continue
        sgs = sorted(x["sequence_gradients_s"] for x in xs)
        walls = sorted(x["round_wall"] for x in xs)
        summary["on" if on else "off"] = {
            "n": len(xs), "sg_median": round(st.median(sgs), 3), "sg_min": sgs[0],
            "sg_max": sgs[-1], "sg_spread": round(sgs[-1] / sgs[0], 3), "sg_all": sgs,
            "round_wall_median": round(st.median(walls), 3), "round_wall_all": walls,
            "device_backward_median": round(st.median(x["device_backward_s"] for x in xs), 3),
            "device_evoformer_median": round(st.median(x["device_evoformer_s"] for x in xs), 3),
            "device_extra_msa_median": round(st.median(x["device_extra_msa_s"] for x in xs), 3),
            "device_share_median": round(st.median(x["device_share_of_sg"] for x in xs), 4),
            "aiclk_min": min((x["aiclk_min"] for x in xs if x["aiclk_min"]), default=None),
            "aiclk_med_median": st.median([x["aiclk_med"] for x in xs if x["aiclk_med"]] or [0])}
    if "on" in summary and "off" in summary:
        # The device metric first: it is the one the lever acts on and the one ten rounds on this
        # tree can actually resolve (CV 3.27 % against the wall's 5.80 %).
        summary["ratio_device_evo_off_over_on"] = round(
            summary["off"]["device_evoformer_median"]
            / max(summary["on"]["device_evoformer_median"], 1e-9), 4)
        summary["device_evo_separated"] = (
            max(x["device_evoformer_s"] for x in timed if x["levers_on"])
            < min(x["device_evoformer_s"] for x in timed if not x["levers_on"]))
        summary["ratio_sg_off_over_on"] = round(summary["off"]["sg_median"]
                                                / summary["on"]["sg_median"], 4)
        summary["ratio_round_off_over_on"] = round(summary["off"]["round_wall_median"]
                                                   / summary["on"]["round_wall_median"], 4)
        summary["separated"] = summary["on"]["sg_max"] < summary["off"]["sg_min"]
        # Paired, to absorb the box's drift: each OFF round against the ON that followed it.
        seq = [(x["levers_on"], x["sequence_gradients_s"]) for x in timed]
        pairs = [(a[1] - b[1]) for a, b in zip(seq, seq[1:]) if a[0] is False and b[0] is True]
        summary["paired_off_minus_on_s"] = [round(d, 3) for d in pairs]
        if pairs:
            summary["paired_mean_s"] = round(sum(pairs) / len(pairs), 3)
    return rows, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=28,
                    help="13 reps per arm plus two warm-up rounds: what the round wall needs to "
                         "resolve the predicted 1.071x at 80 %% power (perf/bcx_bwbytes/power.py). "
                         "The device metric needs 3 per arm, so a short run still answers the "
                         "device question and only the wall goes unresolved")
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--precision", action="store_true",
                    help="the ON arm also takes the two precision levers; they move the gradient "
                         "and are graded as a stack against float64 before they count")
    ap.add_argument("--tree-rows", type=int, default=256,
                    help="LEADING_SUM_TREE_ROWS in the ON arm; set from perf/bcx_bwbytes/probe.py")
    ap.add_argument("--params", default="/home/ttuser/bcx_e2e/af2_params")
    ap.add_argument("--no-extra-msa", action="store_true")
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
    campaign.AlphaFoldDesignModel = functools.partial(T.TTBioAlphaFoldDesignModel,
                                                      trunk="device")

    M.CLOCK = M.Clock(1.0)
    M.CLOCK.start()
    stamp = {"host": os.uname().nodename, "card": os.environ.get("TT_VISIBLE_DEVICES"),
             "pci": M.CLOCK.pci, "sysfs": M.CLOCK.path, "commit": git_head(),
             "seed": args.seed, "rounds_requested": args.rounds,
             "tree_rows_on": args.tree_rows, "precision_arm": args.precision,
             "extra_msa_on_device": not args.no_extra_msa,
             "omp": os.environ.get("OMP_NUM_THREADS"),
             "xla_flags": os.environ.get("XLA_FLAGS"),
             "started_utc": time.strftime("%FT%TZ", time.gmtime()),
             "loadavg_start": os.getloadavg(), "nproc": os.cpu_count(), "project": project}
    print(json.dumps(stamp, indent=1), flush=True)

    import afgrad as _A
    import splice
    from splice import EvoformerOnDevice, ExtraMsaOnDevice, evoformer_on_device

    apply = install_levers(args.tree_rows, args.precision)
    mt = ABMeter(args.rounds, apply)
    M.install(mt, splice, T.TTBioAlphaFoldDesignModel, trajectory, seqopt)
    for cls, tag in ((EvoformerOnDevice, "evo"), (ExtraMsaOnDevice, "extra")):
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

    t0 = time.time()
    stopped = None
    with evoformer_on_device(evo, extra_msa=extra):
        try:
            campaign.run_campaign(settings, project, af2_weights=args.params,
                                  mpnn_weights=os.path.join(B.BC2, "bindcraft", "weights",
                                                            "proteinmpnn", "weights_neutral"),
                                  max_trajectories=1)
        except M.StopAfterRounds as stop:
            stopped = str(stop)
        finally:
            M.CLOCK.stop()

    rows, summary = analyse(M.EVENTS, M.CLOCK.samples)
    stamp.update({"wall_seconds": round(time.time() - t0, 2), "stopped": stopped,
                  "evo_calls": dict(evo.calls),
                  "extra_calls": dict(extra.calls) if extra else None,
                  "loadavg_end": os.getloadavg(), "lever_counts": dict(COUNT)})
    blob = {"stamp": stamp, "rounds": rows, "summary": summary}
    pathlib.Path(project, "round_ab.json").write_text(json.dumps(blob, indent=1))
    print(json.dumps({"summary": summary, "lever_counts": dict(COUNT)}, indent=1), flush=True)
    served = COUNT["reblock_back:on"] + COUNT["reblock_fwd:on"] + COUNT["tree:on"]
    off = COUNT["reblock_back:off"] + COUNT["reblock_fwd:off"] + COUNT["tree:off"]
    if args.precision:
        # The precision arm fires in a dtype, not in a call count, so the control it needs is
        # that the softmax backward ran in a DIFFERENT dtype on the two arms. Counting calls
        # would read identically either way, which is exactly how an inert lever passes.
        dts = {k: v for k, v in COUNT.items() if k.startswith("softmax_bw:")}
        on_dt = {k.split(":")[-1] for k in dts if ":on:" in k}
        off_dt = {k.split(":")[-1] for k in dts if ":off:" in k}
        if on_dt and off_dt and on_dt == off_dt:
            raise RuntimeError(f"the precision arm did not change the softmax backward's dtype: "
                               f"{dts} -- the lever is wired but inert")
        served += sum(v for k, v in dts.items() if ":on:" in k)
    if any(r["levers_on"] for r in rows if r["round"] > 2) and served == 0:
        raise RuntimeError(f"ON rounds ran and served nothing: {dict(COUNT)} -- the lever is "
                           f"inert at this n, which is a RESULT, not a measurement of the round")
    if off:
        raise RuntimeError(f"OFF rounds served {off} lever calls: the arm does not separate")
    print(f"served {served} lever calls on the ON arm, {off} on the OFF arm", flush=True)


if __name__ == "__main__":
    main()
