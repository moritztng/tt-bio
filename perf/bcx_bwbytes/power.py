#!/usr/bin/env python3
"""bcx-bwbytes: can the round even resolve this lever, and on which metric.

The predicted win is small, so the first question is not whether to run the A/B but whether the
A/B could see it. `bcx-tmplseam` measured twelve rounds on the tree this branch is reset onto and
recorded, per round, both the wall and the device seconds inside `sequence_gradients`. That is
enough to answer three things without a card:

  1. the DEVICE SHARE of the round on this tree, measured rather than inherited -- the row exists
     because a 20.1 % share from one tree was quoted against another for five passes
  2. the spread of each metric, and therefore how many reps per arm are needed to resolve an
     effect of a given size
  3. what the levers are predicted to buy on the ROUND once the device share is the measured one
     and the backward is only part of what the card does per round

On (3) the correction matters and it cuts the prediction down. `reach.py` models a 4+48 gradient
step's BACKWARD. `device_evoformer_s` is the whole of what the card does for the Evoformer inside
one round -- the primal forward, the taped forward and the backward. By bcx-bytes' own byte
census those are 9.40, 9.40 and 39.39 GB per block at n=256, so the backward is 67.7 % of it, and
a lever that only touches the backward reaches only that fraction of the round's device time.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import statistics as st
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent

#: bcx-bytes, census.json: taped forward / backward DRAM bytes per Evoformer block at n=256. The
#: primal forward is an untaped run of the same graph, so it is charged the taped forward's bytes.
FWD_GB, BWD_GB = 9.40, 39.394


def cv(xs):
    m = st.mean(xs)
    return st.pstdev(xs) / m if len(xs) > 1 and m else 0.0


def reps_needed(sigma_rel, effect_rel, power_z=2.80):
    """Reps per arm for a two-sample difference of means at ~80 % power, 5 % two-sided.

    n = 2 * (z_a/2 + z_b)^2 * sigma^2 / delta^2, with (1.96 + 0.84) = 2.80. Relative units, so
    sigma and delta are both fractions of the mean and the scale cancels.
    """
    if effect_rel <= 0:
        return None
    return math.ceil(2 * power_z ** 2 * (sigma_rel / effect_rel) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", default=str(ROOT / "perf/bcx_tmplseam/runs/round_ab_artifacts"
                                                   "/round_ab.json"))
    ap.add_argument("--scale", default=str(HERE / "reach_scale.json"))
    ap.add_argument("--warmup", type=int, default=2, help="rounds 1-2 carry the JAX compiles")
    ap.add_argument("--out", default=str(HERE / "power.json"))
    args = ap.parse_args()

    d = json.loads(pathlib.Path(args.rounds).read_text())
    rows = [r for r in d["rounds"] if r["round"] > args.warmup]
    sc = json.loads(pathlib.Path(args.scale).read_text())

    metrics = {"round_wall": [r["round_wall"] for r in rows],
               "sequence_gradients_s": [r["sequence_gradients_s"] for r in rows],
               "device_evoformer_s": [r["device_evoformer_s"] for r in rows],
               "device_extra_msa_s": [r["device_extra_msa_s"] for r in rows]}
    share = [r["device_share_of_sg"] for r in rows]

    dev_s = st.median(metrics["device_evoformer_s"]) + st.median(metrics["device_extra_msa_s"])
    sg = st.median(metrics["sequence_gradients_s"])
    bwd_frac = BWD_GB / (2 * FWD_GB + BWD_GB)

    out = {"source": args.rounds, "n_timed_rounds": len(rows),
           "aiclk_med": sorted({r["aiclk_med"] for r in rows}),
           "load1": [min(r["load1"] for r in rows), max(r["load1"] for r in rows)],
           "device_share_of_sg": {"median": round(st.median(share), 4),
                                  "min": min(share), "max": max(share)},
           "medians_s": {k: round(st.median(v), 3) for k, v in metrics.items()},
           "spread_max_over_min": {k: round(max(v) / min(v), 3) for k, v in metrics.items()},
           "cv": {k: round(cv(v), 4) for k, v in metrics.items()},
           "backward_share_of_device": round(bwd_frac, 4),
           "device_seconds_per_round": round(dev_s, 3),
           "sequence_gradients_s": round(sg, 3)}

    # The levers, as a fraction of the BACKWARD's bytes at n=224, from reach_scale.json
    ev = sc["evo"]
    lev_gb = sum(ev["levers"][k]["at"]["224"] for k in ev["levers"])
    tot_gb = ev["totals"]["224"]
    per = sc["fused_kernel_ceiling"]["per_kernel"]["evo"]
    sm_gb = per["fused_softmax"]["removed_GB"]["224"]
    sm_wheel = per["fused_softmax"]["removed_GB_wheel"]["224"]
    ln_gb = per["fused_layernorm"]["removed_GB"]["224"]
    # `plus_wheel_softmax` is the one a card can test TODAY: main's SOFTMAX_BW_FUSED turned on
    # with bf16 operands, no kernel written. It writes bf16, so it also pays the widen-dx that
    # `backward()` inserts, which is why it is below the custom-kernel line rather than equal to
    # it. Ordered so the cheapest testable scenario reads first.
    out["levers"] = {}
    for name, gb in (("precision_stack", lev_gb),
                     ("plus_wheel_softmax_no_kernel", lev_gb + sm_wheel),
                     ("plus_fused_softmax", lev_gb + sm_gb),
                     ("plus_both_fused_kernels", lev_gb + sm_gb + ln_gb)):
        f_bwd = gb / tot_gb                      # share of the backward's bytes
        f_dev = f_bwd * bwd_frac                 # share of the round's device time
        f_round = f_dev * st.median(share)       # share of the round
        r = 1.0 / (1.0 - f_round)
        out["levers"][name] = {
            "removed_GB_per_evo_block_n224": round(gb, 3),
            "share_of_backward_bytes": round(f_bwd, 4),
            "share_of_round_device_time": round(f_dev, 4),
            "round_speedup": round(r, 4),
            "reps_per_arm": {k: reps_needed(out["cv"][k],
                                            f_dev if k.startswith("device") else f_round)
                             for k in metrics}}

    pathlib.Path(args.out).write_text(json.dumps(out, indent=1))

    print(f"{out['n_timed_rounds']} timed rounds, AICLK {out['aiclk_med']}, "
          f"load {out['load1'][0]}-{out['load1'][1]}, n=211 bucketed to 224")
    print(f"\nDEVICE SHARE of sequence_gradients on this tree: median "
          f"{out['device_share_of_sg']['median']:.3f} "
          f"({out['device_share_of_sg']['min']:.3f}-{out['device_share_of_sg']['max']:.3f})")
    print(f"   -> deleting ALL device time is {1 / (1 - out['device_share_of_sg']['median']):.2f}x, "
          f"which is the ceiling every lever here works against")
    print("\nper-metric variability over the same ten rounds:")
    for k in metrics:
        print(f"   {k:24s} median {out['medians_s'][k]:7.3f} s   spread "
              f"{out['spread_max_over_min'][k]:5.3f}x   CV {out['cv'][k] * 100:5.2f} %")
    print(f"\nthe backward is {bwd_frac * 100:.1f} % of the card's Evoformer work per round "
          f"(primal fwd {FWD_GB} + taped fwd {FWD_GB} + bwd {BWD_GB} GB per block), so a "
          f"backward-only lever reaches only that fraction")
    for name, v in out["levers"].items():
        print(f"\n{name}: removes {v['share_of_backward_bytes'] * 100:.1f} % of the backward's "
              f"bytes = {v['share_of_round_device_time'] * 100:.1f} % of the round's device time "
              f"-> {v['round_speedup']:.3f}x on the round")
        print("   reps per arm to resolve it at 80 % power, 5 % two-sided:")
        for k, n in v["reps_per_arm"].items():
            print(f"     {k:24s} {n}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
