#!/usr/bin/env python3
"""Samples/sec, per box, per dollar and per watt for OpenFold3 training: TT vs H200.

Every input is either read out of a banked artifact in this repo or is a list price / power
figure with its source named in SOURCES below. No number here is measured by this script --
it is arithmetic over numbers other rows measured, and it prints its own inputs so the
arithmetic can be checked without re-running anything.

A SAMPLE is one training example: one crop of 384 tokens at batch 1. It is NOT a diffusion
sample; the TT step ran 4 of those and upstream's recipe runs 48, and that asymmetry is
carried through as a separate term rather than folded into the headline.
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent.parent

SOURCES = {
    "tt_step_s": "perf/of3t_stepfloor/out/step_rekey_b_384.json rep 2, AICLK median 1350 MHz "
                 "over 578 DURING samples, tt-quietbox2 dev3, crop 384, batch 1, 4 diffusion "
                 "samples, full taped step (of3t-gpugap HEADLINE)",
    "h200_step_s": "state/of3t/EVIDENCE.md:89 -- H200 at 1980 MHz, complete Lightning step, "
                   "bf16-mixed, batch 1, crop 384, 48 diffusion samples. n=2 steady samples",
    "tt_box": "tenstorrent.com/hardware/tt-quietbox -- TT-QuietBox 2 (Blackhole), $9,999, "
              "four Blackhole chips / 480 Tensix / 128 GB (docs.tenstorrent.com/systems/"
              "quietbox/quietbox-bh-2/setup.html)",
    "tt_box_w": "docs.tenstorrent.com/systems/quietbox/quietbox-bh-2/setup.html -- 'up to 1300W "
                "while running high-load models'; PSU rated 1600 W",
    "tt_chip_price": "p150a $1,399 list (tenstorrent.com/hardware/blackhole). The p300c is not "
                     "sold separately and has no published price; knowledge/marketing/claims.md "
                     "already prices one p300c processor as a p150a, same 120 Tensix / 32 GB / "
                     "512 GB/s. Convention reused, not invented here",
    "dgx_price": "$410,000 list for 8 GPUs (the figure in this row's brief). Independent quotes "
                 "found range $410k-$450k+; the lowest is used, which flatters the GPU",
    "dgx_w": "NVIDIA DGX H200 datasheet -- 10.2 kW system max",
    "dp_2chip": "perf/train_d_dp/out/scale_rank8.json and scale_rank128.json -- the only "
                "multi-chip training scaling measured on this fleet, 2x p150a at 1350 MHz",
    "elec": "$0.0871/kWh and 4-year straight-line at 8766 h/year -- the TCO convention already "
            "used in knowledge/marketing/claims.md:54",
    "params": "perf/of3t_stepfloor/out/step_rekey_b_384.json params.elements",
}

# ---- measured inputs ------------------------------------------------------------------
step = json.loads((REPO / "perf/of3t_stepfloor/out/step_rekey_b_384.json").read_text())
rep = next(r for r in step["reps"] if r["step_s"] == 466.702)
TT_STEP_S = rep["step_s"]
warm = [r["step_s"] for r in json.loads(
            (REPO / "perf/of3t_stepfloor/out/step_rekey_384.json").read_text())["reps"]
        if not r["cold"]] + [r["step_s"] for r in step["reps"] if not r["cold"]]
TT_STEP_WORST = max(warm)
ELEMENTS = step["params"]["elements"]
UNPORTED_S = rep["losses_s"] + rep["optimizer_s"]

H200_STEP_S = (7.0, 8.0)

dp8 = json.loads((REPO / "perf/train_d_dp/out/scale_rank8.json").read_text())["arms"]
dp128 = json.loads((REPO / "perf/train_d_dp/out/scale_rank128.json").read_text())["arms"]
EFF2 = sorted((dp128["2"]["examples_per_s_median"] / dp128["1"]["examples_per_s_median"] / 2,
               dp8["2"]["examples_per_s_median"] / dp8["1"]["examples_per_s_median"] / 2))
SHM_BPS = dp128["2"]["comm_bytes"] / dp128["2"]["median_transfer_s"]

# ---- prices and power -----------------------------------------------------------------
TT_BOX_CHIPS, TT_BOX_USD, TT_BOX_KW = 4, 9_999.0, 1.300
TT_CHIP_USD, TT_CHIP_KW = 1_399.0, 0.300
DGX_GPUS, DGX_USD, DGX_KW = 8, 410_000.0, 10.2
H200_KW = 0.700
ELEC_USD_KWH, LIFE_H = 0.0871, 4 * 8766

SOFT_FIX = (6.0, 7.0)          # BACKWARD.md: 58-67x total over ~9x hardware


def per_h(step_s):
    return 3600.0 / step_s


def tco_usd_h(capex, kw):
    return capex / LIFE_H + kw * ELEC_USD_KWH


def box(rate_chip_h, chips, eff):
    return rate_chip_h * chips * eff


def rng(f, *args):
    v = [f(*a) for a in args]
    return (min(v), max(v))


out = {"definition": "one sample = one training example = one crop of 384 tokens, batch 1. "
                     "Not a diffusion sample.",
       "sources": SOURCES,
       "inputs": {"tt_step_s": TT_STEP_S, "tt_step_s_worst_warm": TT_STEP_WORST,
                  "h200_step_s": H200_STEP_S, "tt_unported_host_s": round(UNPORTED_S, 3),
                  "params_elements": ELEMENTS,
                  "dp_2chip_efficiency": [round(e, 4) for e in EFF2],
                  "shm_all_reduce_B_per_s": round(SHM_BPS)}}

# ---- SAMPLES --------------------------------------------------------------------------
tt_chip = per_h(TT_STEP_S)
tt_chip_worst = per_h(TT_STEP_WORST)
h200_gpu = (per_h(H200_STEP_S[1]), per_h(H200_STEP_S[0]))
tt_box = (box(tt_chip, TT_BOX_CHIPS, EFF2[0]), box(tt_chip, TT_BOX_CHIPS, 1.0))
dgx_box = (h200_gpu[0] * DGX_GPUS, h200_gpu[1] * DGX_GPUS)

out["samples_per_hour"] = {
    "tt_chip": round(tt_chip, 4), "tt_chip_worst_warm_rep": round(tt_chip_worst, 4),
    "h200_gpu": [round(x, 3) for x in h200_gpu],
    "tt_box_4chip": [round(x, 3) for x in tt_box],
    "dgx_h200_8gpu": [round(x, 2) for x in dgx_box],
}
out["samples_per_sec"] = {
    "tt_chip": tt_chip / 3600, "h200_gpu": [x / 3600 for x in h200_gpu],
    "tt_box_4chip": [x / 3600 for x in tt_box], "dgx_h200_8gpu": [x / 3600 for x in dgx_box],
}

# ---- BOXGAP ---------------------------------------------------------------------------
def gap(tt, gpu):
    return (gpu[0] / tt[1], gpu[1] / tt[0])


out["gap"] = {
    "chip_today": [round(h200_gpu[0] / tt_chip, 1), round(h200_gpu[1] / tt_chip, 1)],
    "box_today": [round(x, 1) for x in gap(tt_box, dgx_box)],
}
fixed_chip = [per_h(TT_STEP_S / f) for f in SOFT_FIX]
fixed_box = (box(min(fixed_chip), TT_BOX_CHIPS, EFF2[0]),
             box(max(fixed_chip), TT_BOX_CHIPS, 1.0))
out["gap"]["chip_after_6_7x"] = [round(h200_gpu[0] / max(fixed_chip), 1),
                                 round(h200_gpu[1] / min(fixed_chip), 1)]
out["gap"]["box_after_6_7x"] = [round(x, 1) for x in gap(fixed_box, dgx_box)]
out["samples_per_hour"]["tt_box_4chip_after_6_7x"] = [round(x, 2) for x in fixed_box]

# ---- PERDOLLAR / PERWATT --------------------------------------------------------------
def block(name, rate, capex, kw):
    lo, hi = (rate, rate) if isinstance(rate, float) else (rate[0], rate[1])
    t = tco_usd_h(capex, kw)
    return {name: {
        "samples_per_hour": [round(lo, 3), round(hi, 3)],
        "capex_usd": capex, "kW": kw,
        "samples_per_hour_per_1k_usd_list": [round(lo / (capex / 1000), 4),
                                             round(hi / (capex / 1000), 4)],
        "samples_per_hour_per_kW": [round(lo / kw, 2), round(hi / kw, 2)],
        "tco_usd_per_hour": round(t, 4),
        "samples_per_tco_dollar": [round(lo / t, 2), round(hi / t, 2)]}}


perf = {}
perf.update(block("tt_quietbox2", tt_box, TT_BOX_USD, TT_BOX_KW))
perf.update(block("tt_quietbox2_after_6_7x", fixed_box, TT_BOX_USD, TT_BOX_KW))
perf.update(block("dgx_h200", dgx_box, DGX_USD, DGX_KW))
perf.update(block("tt_chip_p150a_priced", tt_chip, TT_CHIP_USD, TT_CHIP_KW))
perf.update(block("h200_gpu_alone", h200_gpu, DGX_USD / DGX_GPUS, H200_KW))
out["perdollar"] = perf

d, t = perf["dgx_h200"], perf["tt_quietbox2"]
tf = perf["tt_quietbox2_after_6_7x"]
out["perdollar"]["ratio_gpu_advantage"] = {
    "per_1k_usd_list_today": [round(d["samples_per_hour_per_1k_usd_list"][0]
                                    / t["samples_per_hour_per_1k_usd_list"][1], 2),
                              round(d["samples_per_hour_per_1k_usd_list"][1]
                                    / t["samples_per_hour_per_1k_usd_list"][0], 2)],
    "per_1k_usd_list_after_6_7x": [round(d["samples_per_hour_per_1k_usd_list"][0]
                                         / tf["samples_per_hour_per_1k_usd_list"][1], 2),
                                   round(d["samples_per_hour_per_1k_usd_list"][1]
                                         / tf["samples_per_hour_per_1k_usd_list"][0], 2)],
    "per_kW_today": [round(d["samples_per_hour_per_kW"][0] / t["samples_per_hour_per_kW"][1], 2),
                     round(d["samples_per_hour_per_kW"][1] / t["samples_per_hour_per_kW"][0], 2)],
    "per_kW_after_6_7x": [round(d["samples_per_hour_per_kW"][0]
                                / tf["samples_per_hour_per_kW"][1], 2),
                          round(d["samples_per_hour_per_kW"][1]
                                / tf["samples_per_hour_per_kW"][0], 2)],
    "per_tco_dollar_today": [round(d["samples_per_tco_dollar"][0] / t["samples_per_tco_dollar"][1], 2),
                             round(d["samples_per_tco_dollar"][1] / t["samples_per_tco_dollar"][0], 2)],
    "per_tco_dollar_after_6_7x": [round(d["samples_per_tco_dollar"][0]
                                        / tf["samples_per_tco_dollar"][1], 2),
                                  round(d["samples_per_tco_dollar"][1]
                                        / tf["samples_per_tco_dollar"][0], 2)],
}

# ---- REPLICATE: what the gradient exchange costs ---------------------------------------
grad_bytes_bf16 = ELEMENTS * 2
out["replicate"] = {
    "grad_bytes_bf16_upper_bound": grad_bytes_bf16,
    "note": "upper bound: 2,660 of 3,152 tensors carry a gradient, so the true payload is at "
            "most every parameter",
    "shm_all_reduce_s": round(grad_bytes_bf16 / SHM_BPS, 3),
    "share_of_step_today": round(grad_bytes_bf16 / SHM_BPS / TT_STEP_S * 100, 3),
    "share_of_step_after_6_7x": [round(grad_bytes_bf16 / SHM_BPS / (TT_STEP_S / f) * 100, 2)
                                 for f in SOFT_FIX],
}

# ---- the 48-diffusion-sample correction -------------------------------------------------
# of3t-gpugap CEILING: host af3_loss is linear in diffusion samples, 0.455-1.226 s per root
# across the eight reps; AdamW adds 3.0-3.2 s. This is the floor a device side costing
# nothing still pays at upstream's shipped recipe.
out["at_48_diffusion_samples"] = {
    "host_floor_s": [round(0.455 * 48 + 3.0, 1), round(1.226 * 48 + 3.2, 1)],
    "h200_whole_step_s": list(H200_STEP_S),
    "note": "the TT step above ran 4 diffusion samples against upstream's 48; at 48 the host "
            "loss alone exceeds the H200's entire step",
}

# ---- REPLICATE: the host term does NOT replicate ----------------------------------------
# Four chips in one QuietBox share ONE 8-core Ryzen. The device work replicates across chips;
# the unported host work (af3_loss numpy + AdamW fp32 masters) lands on that one CPU four
# times over. Whether that serialises is unmeasured -- it depends on numpy's own threading --
# so both bounds are given rather than one picked.
out["replicate"]["host_contention"] = {
    "host_s_per_rank_at_4_samples": round(UNPORTED_S, 3),
    "host_s_per_rank_at_48_samples": out["at_48_diffusion_samples"]["host_floor_s"],
    "four_ranks_if_perfectly_parallel_s": out["at_48_diffusion_samples"]["host_floor_s"],
    "four_ranks_if_fully_serialised_s": [round(4 * x, 1)
                                         for x in out["at_48_diffusion_samples"]["host_floor_s"]],
    "post_fix_step_s": [round(TT_STEP_S / f, 1) for f in SOFT_FIX],
    "note": "at 48 diffusion samples the host term already exceeds a post-fix step even fully "
            "parallel, so on a 4-chip box the CPU, not the chips, sets samples/sec",
}

(HERE / "out").mkdir(exist_ok=True)
(HERE / "out" / "throughput.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
