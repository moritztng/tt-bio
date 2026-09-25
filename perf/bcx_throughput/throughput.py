#!/usr/bin/env python3
"""BindCraft 2 design throughput: per chip, per box, per dollar, per watt.

Arithmetic only. Every input below is a measurement another row banked, or a list price with a
named public source. No card is opened, no GPU is rented, nothing is re-measured here. Run it to
re-check any number in `state/bcx-throughput.md`:

    python3 perf/bcx_throughput/throughput.py

A trajectory is one BindCraft 2 design cycle against hPDL1 chain A (115 aa) with a 146 aa binder,
the shipped five-model `multimer_v3` pool of `examples/pdl1.json`: 125 gradient rounds, then 15
forward-only mutate steps, then ProteinMPNN redesign and the validation refold ensemble.

Two axes are computed and they are NOT interchangeable:

  * GRADIENT PHASE -- the 125 rounds alone. Both sides measured, matched on target, binder length
    and model pool. This is `bcx-gpuref`'s phase-matched axis.
  * WHOLE CYCLE -- the phase plus the ProteinMPNN/validation tail. The tail is what "accepted
    designs per hour" divides into, so it is the axis a buyer cares about.

The 15 mutate steps are carried by neither side's TT figure. On the H200 they are 6.03 s p50,
5.4 % of a cycle, so leaving them out understates the TT cycle by a few percent and flatters us.
"""
import json
from pathlib import Path

# --------------------------------------------------------------------------------------------
# INPUTS. Each is (value, source). Nothing below this block is an input.
# --------------------------------------------------------------------------------------------
IN = {
    # -- Tenstorrent, Blackhole p300c on tt-quietbox2, AICLK 1350 MHz sampled DURING every fold.
    "tt_phase_s_fast": (6621.2, "state/bcx/ACCEPT.md:116 -- arm `step288` draw 9f676a43, "
                                "6621.2 s / 125 = 53.0 s/step, qb2 p300c, AICLK 1350 during"),
    "tt_phase_s_slow": (8069.9, "state/bcx/NUMBERS.md:38 and bcx-gpuref -- arm `pool_full` draw "
                                "3150367d, 8069.9 s / 125 = 64.6 s/step, qb2 p300c, AICLK 1350 "
                                "during. This is the end `bcx-gpuref` phase-matched 89.7x on"),
    "tt_phase_s_clean": (7250.0, "state/bcx/ACCEPT.md:116 -- the checkpoint-clean arm in flight at "
                                 "58 s/step over its first 61 steps, x125. Corroboration that the "
                                 "band is not an artifact of the two chimera-trunk arms"),
    # -- the unported tail. Our device arm runs ProteinMPNN redesign and the validation ensemble on
    # host JAX by design (state/bcx/VALIDATION-SEAM-SPLIT.md: "Default for any acceptance
    # measurement: host JAX"), so the reference's own tail IS our tail's code path.
    "tt_tail_s": (1253.0, "state/bcx-shipped.md:77 -- BindCraft 2's own JAX, pc, seed 0, "
                          "trajectory 1 at binder 146: 1,253 s for 5 candidates scored, ~250 s "
                          "each, candidate lines 07:28:06Z to 07:44:35Z. Host CPU wall, not chip "
                          "time, and measured on pc rather than on qb2's Ryzen"),

    # -- NVIDIA H200, 1980 MHz on all 253 samples, 98-100 % util, nothing else on the card.
    "h200_phase_s": (89.93, "state/bcx-gpuref.md -- gradient phase p50 over the 13 trajectories "
                            "that ran all 125 steps, binder 146, H200 at 1980 MHz"),
    "h200_cycle_s": (112.3, "state/bcx-gpuref.md -- whole cycle p50, n=6 (gradient + mutate + "
                            "ProteinMPNN redesign + validation)"),
    "h200_step_s": (0.6958, "state/bcx-gpuref.md -- warm gradient step p50, n=1698, sd/mean 1.41 %"),

    # -- acceptance. A property of the design algorithm, not of the silicon.
    "h200_accepted": (4, "state/bcx-gpuref.md -- 4 accepted designs from 14 trajectories, all "
                         "four validated as real two-chain complexes at the requested lengths"),
    "h200_trajectories": (14, "same run"),

    # -- packing.
    "tt_pack2_idle": (1.91, "state/bcx-concurrent.md run B -- 2 arms on 2 chips of one QuietBox, "
                            "4.6 % cost per arm, AICLK 1350 median on every card"),
    "tt_pack2_loaded": (1.55, "state/bcx-concurrent.md run A -- same 2 arms with ~10 threads of "
                              "other work on the host, 28.9 % cost per arm"),
    "tt_arm_threads": (1.9, "state/bcx-concurrent.md -- an arm draws 1.78-1.9 of qb2's 16 threads"),
    "gpu_pack_workers": (7, "state/bcx/GPUDOC.md:100 -- BC2 packs up to 7 design workers per card, "
                            "`max_workers_per_gpu` default 8"),
    "gpu_pack_gain": (1.79, "state/bcx/GPUDOC.md:102 -- BC2's own measured packing gain on a "
                            "GH200, 3620 s at 1 worker against 2020 s at 7, whole cycles. It "
                            "fills ProteinMPNN and validation gaps rather than multiplying "
                            "gradient design, so it is 1.00x on the gradient-phase axis"),

    # -- boxes, list prices and power, public.
    "tt_box_chips": (4, "TT-QuietBox 2, 4x Blackhole"),
    "tt_box_usd": (9999, "knowledge/reference/tenstorrent-company.md:40 and "
                         "tenstorrent.com/hardware/tt-quietbox"),
    "tt_box_kw": (1.3, "docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/setup.html -- up to "
                       "1300 W under high-load models; the PSU is rated 1600 W"),
    "gpu_box_gpus": (8, "DGX H200, 8x H200 SXM"),
    "gpu_box_usd": (410000, "knowledge/marketing/claims.md:236 -- DGX H200 $410,000 list. Quotes "
                            "range $410k-$450k+; the lowest is used, which flatters the GPU"),
    "gpu_box_kw": (10.2, "knowledge/marketing/claims.md:236 -- DGX H200 10.2 kW system max"),

    # -- TCO convention, unchanged from knowledge/marketing/claims.md:54 and of3t-throughput.
    "tco_years": (4, "straight-line over 4 years"),
    "tco_h_per_year": (8766, "hours per year"),
    "usd_per_kwh": (0.0871, "US EIA industrial average, May 2026"),

    # -- the hardware floor, so the software residue is not asserted.
    "hw_ratio_lo": (8.5, "state/of3t/BACKWARD.md -- Blackhole 115.685 TFLOP/s measured against "
                         "H200 ~989 TFLOP/s BF16 dense: ~8.5x on compute"),
    "hw_ratio_hi": (11.0, "state/of3t/BACKWARD.md -- ~440 GB/s against 4.8 TB/s: ~11x on "
                          "bandwidth, which is the side this workload sits on"),
}
V = {k: v[0] for k, v in IN.items()}
H = 3600.0


def rng(lo, hi):
    return (min(lo, hi), max(lo, hi))


out = {"inputs": {k: {"value": v[0], "source": v[1]} for k, v in IN.items()}}

# -- RATE, per chip ------------------------------------------------------------------------------
tt_phase = rng(V["tt_phase_s_fast"], V["tt_phase_s_slow"])
tt_cycle = (tt_phase[0] + V["tt_tail_s"], tt_phase[1] + V["tt_tail_s"])
h200_tail = V["h200_cycle_s"] - V["h200_phase_s"]

rate = {
    "tt_chip_phase_per_h": rng(H / tt_phase[1], H / tt_phase[0]),
    "tt_chip_cycle_per_h": rng(H / tt_cycle[1], H / tt_cycle[0]),
    "h200_phase_per_h": H / V["h200_phase_s"],
    "h200_cycle_per_h": H / V["h200_cycle_s"],
    "tt_cycle_s": tt_cycle,
    "h200_tail_s": h200_tail,
    "tail_ratio": V["tt_tail_s"] / h200_tail,
    "tt_tail_share_today": rng(V["tt_tail_s"] / tt_cycle[0], V["tt_tail_s"] / tt_cycle[1]),
    "h200_tail_share": h200_tail / V["h200_cycle_s"],
    "chip_ratio_phase": rng(tt_phase[0] / V["h200_phase_s"], tt_phase[1] / V["h200_phase_s"]),
    "chip_ratio_cycle": rng(tt_cycle[0] / V["h200_cycle_s"], tt_cycle[1] / V["h200_cycle_s"]),
    "chip_ratio_step": rng(V["tt_phase_s_fast"] / 125 / V["h200_step_s"],
                           V["tt_phase_s_slow"] / 125 / V["h200_step_s"]),
    "accept_rate_h200": V["h200_accepted"] / V["h200_trajectories"],
}

# -- PACKING -------------------------------------------------------------------------------------
# The 4-arm projection. Upper end follows bcx-concurrent's own mechanism: cost per arm tracks TOTAL
# busy threads on the box, and 4 arms alone is 4 x 1.9 = 7.6 of 16, BELOW run B's ~8.8, where the
# measured cost was 4.6 %. Lower end refuses to extrapolate at all and charges every one of the 4
# arms run A's loaded-box cost. 4 arms have never been run; 3 have never been run either.
pack4_hi = V["tt_box_chips"] / (2 / V["tt_pack2_idle"])          # 4 x the per-arm efficiency at N=2 idle
pack4_lo = V["tt_box_chips"] / (2 / V["tt_pack2_loaded"])        # ... at N=2 on a loaded box
packing = {
    "tt_pack4": rng(pack4_lo, pack4_hi),
    "tt_pack4_efficiency": rng(pack4_lo / V["tt_box_chips"], pack4_hi / V["tt_box_chips"]),
    "tt_arm_threads_at_4": V["tt_arm_threads"] * V["tt_box_chips"],
    "gpu_cost_per_worker": V["gpu_pack_workers"] / V["gpu_pack_gain"],
    "gpu_pack_gain_phase_axis": 1.0,
}

# -- boxes ---------------------------------------------------------------------------------------
tt_box_phase = rng(rate["tt_chip_phase_per_h"][0] * pack4_lo, rate["tt_chip_phase_per_h"][1] * pack4_hi)
tt_box_cycle = rng(rate["tt_chip_cycle_per_h"][0] * pack4_lo, rate["tt_chip_cycle_per_h"][1] * pack4_hi)
# The GPU box is given PERFECT 8-GPU scaling, which flatters it and keeps every gap an upper bound.
gpu_box_phase = rate["h200_phase_per_h"] * V["gpu_box_gpus"]                     # 1 worker/GPU
gpu_box_cycle = rate["h200_cycle_per_h"] * V["gpu_box_gpus"] * V["gpu_pack_gain"]  # 7 workers/GPU
rate.update({"tt_box_phase_per_h": tt_box_phase, "tt_box_cycle_per_h": tt_box_cycle,
             "gpu_box_phase_per_h": gpu_box_phase, "gpu_box_cycle_per_h": gpu_box_cycle,
             "tt_box_accepted_per_h": rng(tt_box_cycle[0] * rate["accept_rate_h200"],
                                          tt_box_cycle[1] * rate["accept_rate_h200"]),
             "gpu_box_accepted_per_h": gpu_box_cycle * rate["accept_rate_h200"]})

# -- the software residue, and what closing it buys end to end -----------------------------------
sw = rng(rate["chip_ratio_phase"][0] / V["hw_ratio_hi"], rate["chip_ratio_phase"][1] / V["hw_ratio_lo"])
# Pair each end of the phase band with the software factor that goes with it, then add the tail,
# which the gradient fix does not touch.
fix_cycle_best = tt_phase[0] / sw[1] + V["tt_tail_s"]
fix_cycle_worst = tt_phase[1] / sw[0] + V["tt_tail_s"]
fix = {
    "software_factor": sw,
    "post_fix_chip_floor": (V["hw_ratio_lo"], V["hw_ratio_hi"]),
    "post_fix_cycle_s": rng(fix_cycle_best, fix_cycle_worst),
    "post_fix_tail_share": rng(V["tt_tail_s"] / fix_cycle_worst, V["tt_tail_s"] / fix_cycle_best),
    "end_to_end_gain": rng(tt_cycle[0] / fix_cycle_best, tt_cycle[1] / fix_cycle_worst),
    "post_fix_tt_box_cycle_per_h": rng(H / fix_cycle_worst * pack4_lo, H / fix_cycle_best * pack4_hi),
}

# -- the tail lever, labelled as the transfer it is ---------------------------------------------
# `bcx-backend`'s `TrunkPool.require()` already lets the device pool grow to hold the validation
# checkpoint, so the tail CAN run on card. Nobody has timed it there. The only transfer available
# is the gradient step's own device-vs-same-CPU ratio, 2.6x (state/bcx/NUMBERS.md), applied to a
# forward-only stage it was not measured on. Treated as a projection, never as a measurement.
tail_oncard = V["tt_tail_s"] / 2.6
fix["tail_oncard_s_projected"] = tail_oncard
fix["tail_oncard_gain_today"] = rng(tt_cycle[0] / (tt_phase[0] + tail_oncard),
                                    tt_cycle[1] / (tt_phase[1] + tail_oncard))
fix["tail_oncard_gain_post_fix"] = rng(fix_cycle_best / (tt_phase[0] / sw[1] + tail_oncard),
                                       fix_cycle_worst / (tt_phase[1] / sw[0] + tail_oncard))
fix["both_levers_cycle_s"] = rng(tt_phase[0] / sw[1] + tail_oncard, tt_phase[1] / sw[0] + tail_oncard)

# -- PERDOLLAR -----------------------------------------------------------------------------------
def tco_per_h(usd, kw):
    return usd / (V["tco_years"] * V["tco_h_per_year"]) + kw * V["usd_per_kwh"]


tt_tco = tco_per_h(V["tt_box_usd"], V["tt_box_kw"])
gpu_tco = tco_per_h(V["gpu_box_usd"], V["gpu_box_kw"])
perdollar = {"tt_tco_usd_per_h": tt_tco, "gpu_tco_usd_per_h": gpu_tco,
             "tco_ratio": gpu_tco / tt_tco}
both = rng(H / fix["both_levers_cycle_s"][1] * pack4_lo, H / fix["both_levers_cycle_s"][0] * pack4_hi)
fix["both_levers_tt_box_cycle_per_h"] = both
boxgap_both = rng(gpu_box_cycle / both[1], gpu_box_cycle / both[0])
for tag, tt, gpu in (("today", tt_box_cycle, gpu_box_cycle),
                     ("post_fix", fix["post_fix_tt_box_cycle_per_h"], gpu_box_cycle),
                     ("post_fix_and_tail", both, gpu_box_cycle)):
    tt_k = rng(tt[0] / (V["tt_box_usd"] / 1000), tt[1] / (V["tt_box_usd"] / 1000))
    gpu_k = gpu / (V["gpu_box_usd"] / 1000)
    tt_w = rng(tt[0] / V["tt_box_kw"], tt[1] / V["tt_box_kw"])
    gpu_w = gpu / V["gpu_box_kw"]
    tt_t = rng(tt[0] / tt_tco, tt[1] / tt_tco)
    gpu_t = gpu / gpu_tco
    perdollar[tag] = {
        "tt_cycles_per_h_per_1k_usd": tt_k, "gpu_cycles_per_h_per_1k_usd": gpu_k,
        "gpu_advantage_per_1k_usd": rng(gpu_k / tt_k[0], gpu_k / tt_k[1]),
        "tt_cycles_per_h_per_kw": tt_w, "gpu_cycles_per_h_per_kw": gpu_w,
        "gpu_advantage_per_kw": rng(gpu_w / tt_w[0], gpu_w / tt_w[1]),
        "tt_cycles_per_tco_usd": tt_t, "gpu_cycles_per_tco_usd": gpu_t,
        "gpu_advantage_per_tco_usd": rng(gpu_t / tt_t[0], gpu_t / tt_t[1]),
    }

# -- BOXGAP --------------------------------------------------------------------------------------
boxgap = {
    "cycle_axis_box_both_levers": boxgap_both,
    "phase_axis_chip": rate["chip_ratio_phase"],
    "phase_axis_box": rng(gpu_box_phase / tt_box_phase[1], gpu_box_phase / tt_box_phase[0]),
    "cycle_axis_chip": rate["chip_ratio_cycle"],
    "cycle_axis_box": rng(gpu_box_cycle / tt_box_cycle[1], gpu_box_cycle / tt_box_cycle[0]),
    "cycle_axis_box_post_fix": rng(gpu_box_cycle / fix["post_fix_tt_box_cycle_per_h"][1],
                                   gpu_box_cycle / fix["post_fix_tt_box_cycle_per_h"][0]),
}

out.update({"rate": rate, "packing": packing, "fix": fix, "perdollar": perdollar, "boxgap": boxgap})


def f(x, n=3):
    if isinstance(x, (tuple, list)):
        return f"{x[0]:.{n}f}-{x[1]:.{n}f}"
    return f"{x:.{n}f}"


print("RATE  (hPDL1 chain A 115 aa, binder 146, shipped five-model multimer_v3 pool)")
print(f"  TT p300c   gradient phase  {f(tt_phase,1)} s   -> {f(rate['tt_chip_phase_per_h'])} traj/h/chip")
print(f"  TT p300c   whole cycle     {f(tt_cycle,1)} s   -> {f(rate['tt_chip_cycle_per_h'])} cycles/h/chip")
print(f"  H200       gradient phase  {V['h200_phase_s']:.2f} s -> {rate['h200_phase_per_h']:.2f} traj/h/GPU")
print(f"  H200       whole cycle     {V['h200_cycle_s']:.2f} s -> {rate['h200_cycle_per_h']:.2f} cycles/h/GPU")
print(f"  TT-QuietBox 2 (4 chips)    phase {f(tt_box_phase)} /h   cycle {f(tt_box_cycle)} /h")
print(f"  DGX H200 (8 GPUs)          phase {gpu_box_phase:.1f} /h   cycle {gpu_box_cycle:.1f} /h (packed 1.79x)")
print(f"  accepted/h/box at {rate['accept_rate_h200']*100:.1f} % acceptance: "
      f"TT {f(rate['tt_box_accepted_per_h'])}   DGX {rate['gpu_box_accepted_per_h']:.1f}")
print(f"  the unported tail: ours {V['tt_tail_s']:.0f} s against theirs {h200_tail:.1f} s = "
      f"{rate['tail_ratio']:.0f}x, {f(rate['tt_tail_share_today'],3)} of our cycle")
print()
print("PACKING")
print(f"  measured  2 chips idle box  {V['tt_pack2_idle']}x    2 chips loaded box {V['tt_pack2_loaded']}x")
print(f"  projected 4 chips           {f(packing['tt_pack4'],2)}x  (efficiency {f(packing['tt_pack4_efficiency'],3)})")
print(f"  GPU  {V['gpu_pack_workers']} workers/card -> {V['gpu_pack_gain']}x, "
      f"so each worker runs {packing['gpu_cost_per_worker']:.2f}x slower; 1.00x on the phase axis")
print()
print("PERDOLLAR")
print(f"  TCO/h   TT {tt_tco:.4f}   DGX {gpu_tco:.4f}   ratio {perdollar['tco_ratio']:.1f}x")
for tag in ("today", "post_fix", "post_fix_and_tail"):
    p = perdollar[tag]
    print(f"  {tag:9s} per $1k list {f(p['gpu_advantage_per_1k_usd'],2)}x   "
          f"per kW {f(p['gpu_advantage_per_kw'],2)}x   per TCO $ {f(p['gpu_advantage_per_tco_usd'],2)}x  (GPU advantage)")
print()
print("FIX")
print(f"  software residue {f(fix['software_factor'],2)}x over a {V['hw_ratio_lo']}-{V['hw_ratio_hi']}x hardware floor")
print(f"  post-fix cycle {f(fix['post_fix_cycle_s'],0)} s, tail is {f(fix['post_fix_tail_share'],3)} of it")
print(f"  end-to-end gain from fixing the gradient phase alone: {f(fix['end_to_end_gain'],2)}x")
print()
print(f"  tail on card (projected, 2.6x transfer): {fix['tail_oncard_s_projected']:.0f} s; "
      f"cycle gain today {f(fix['tail_oncard_gain_today'],2)}x, post-fix {f(fix['tail_oncard_gain_post_fix'],2)}x")
print(f"  both levers: cycle {f(fix['both_levers_cycle_s'],0)} s")
print()
print("BOXGAP")
for k, v in boxgap.items():
    print(f"  {k:26s} {f(v,1)}x")

p = Path(__file__).resolve().parent / "out" / "throughput.json"
p.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(f"\nbanked {p}")
