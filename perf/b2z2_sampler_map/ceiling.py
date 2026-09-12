#!/usr/bin/env python3
"""What the diffusion sampler has to deliver for the Boltz-2 512 aa fold to reach 2x.

Consumes `step_map.json` (this row's per-op map of one step) plus the phase split and the
trunk floor that `b2z2-redteam-ceiling` established, and answers the brief's question 4:
given that the trunk tops out at 1.656x, how much of the 1.954x-2.135x bracket does the
sampler have to carry, and is that reachable at 200 steps.

Every ratio is quoted against the LIVE published cell, 20.079 s
(`site/data/perf-512aa.json`, models[0].cells.p150a.s_per_fold, committed at `fc7fed56`).
No card is opened. Emits `ceiling.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
M = json.loads((HERE / "step_map.json").read_text())

# ---- the denominators, both stated ------------------------------------------------------
CELL = 20.079          # MEASURED, published: site/data/perf-512aa.json @ fc7fed56
PAIRED = 19.6525       # MEASURED, b2z-integrate base arm, card 3, the paired incumbent
RETIRED = 23.841       # the internal re-measure wave 1 quoted; kept only to translate old rows
TARGET = CELL / 2.0    # 10.0395 s

# ---- phase split, MEASURED inside one run (b2z2-redteam-ceiling section 7) ---------------
TRUNK, SAMPLER, REST = 12.2173, 5.2547, 2.1805
assert abs(TRUNK + SAMPLER + REST - PAIRED) < 0.02

# The trunk's movement-free floor with its own 0.355 s of host dispatch held fixed. Wave 1's
# row 4 as b2z2-redteam-ceiling re-derived it: fold 12.124 s = 1.6560x against the cell.
TRUNK_ONLY_FOLD = 12.124
TRUNK_FLOOR = TRUNK_ONLY_FOLD - SAMPLER - REST      # 4.689 s

STEPS = 200
STEP_WALL_MS = 26.40   # MEASURED: the page's own cell, 5.280 s / 200 at fc7fed56

out = {"denominators": {"live_cell_s": CELL, "paired_incumbent_s": PAIRED,
                        "retired_s": RETIRED, "two_x_s": TARGET},
       "phase_split_s": {"trunk": TRUNK, "sampler": SAMPLER, "rest": REST},
       "trunk_floor_s": TRUNK_FLOOR}

# ---- the step, decomposed into terms a lever can actually attack -------------------------
tbl = {t["op"]: t for t in M["op_table"]}
kernel = M["kernel_ms_from_table"]                       # 22.015 ms, MEASURED
gap = STEP_WALL_MS - kernel                              # exposed host dispatch, DERIVED
idle = M["occupancy"]["idle_core_fraction_ms_per_step"]  # 5.341 ms, MEASURED

# `b2z-diffusion-utilization` checked the idle core-seconds against the measured 429.9 GB/s
# and 86 TFLOP/s roofs and found 1.0205 of 1.0682 s/fold survives: re-spreading a program onto
# more cores stops paying once it hits a roof. Use that fraction, not 100 %.
RECOVERABLE = 1.0205 / 1.0682
idle_recoverable = idle * RECOVERABLE

# Ops whose math thread is never resident: pure layout/data movement, no arithmetic at all.
ZERO_MATH = [t for t in M["op_table"] if t["trisc1_residency_pct"] < 1.0]
zm_ms = sum(t["ms_per_step"] for t in ZERO_MATH)
# their share of the idle term, so the two levers are not double-counted
zm_idle = sum(M["occupancy"]["by_op_top"].get(t["op"], 0.0) for t in ZERO_MATH)

# The SDPA head-plumbing tax: what it costs to arrange operands for the 30 SDPA calls.
PLUMBING = ["NlpCreateHeadsDeviceOperation", "ReshapeViewDeviceOperation",
            "SliceDeviceOperation", "TransposeDeviceOperation",
            "NLPConcatHeadsDeviceOperation"]
plumb = sum(tbl[o]["ms_per_step"] for o in PLUMBING if o in tbl)

out["step_ms"] = {
    "wall_measured": STEP_WALL_MS,
    "kernel_measured": kernel,
    "exposed_dispatch_gap_derived": gap,
    "gap_pct_of_wall": 100 * gap / STEP_WALL_MS,
    "grid_underfill_measured": idle,
    "grid_underfill_recoverable": idle_recoverable,
    "perfectly_spread_kernel": kernel - idle,
    "zero_math_layout_ops_ms": zm_ms,
    "zero_math_ops": [t["op"] for t in ZERO_MATH],
    "zero_math_share_of_underfill": zm_idle,
    "sdpa_ms": tbl["SDPAOperation"]["ms_per_step"],
    "sdpa_plumbing_ms": plumb,
    "plumbing_over_sdpa": plumb / tbl["SDPAOperation"]["ms_per_step"],
    "binaryng_ms": tbl["BinaryNgDeviceOperation"]["ms_per_step"],
}

# ---- what the sampler must deliver -------------------------------------------------------
# Trunk at its absolute movement-free floor, `rest` held at measured cost: everything the
# fold still owes 2x has to come out of the sampler.
sampler_budget_2x = TARGET - TRUNK_FLOOR - REST
out["required"] = {
    "sampler_s_for_2x_with_trunk_at_floor": sampler_budget_2x,
    "sampler_ratio_required": SAMPLER / sampler_budget_2x,
    "sampler_ms_per_step_required": 1000 * sampler_budget_2x / STEPS,
    "sampler_seconds_to_remove": SAMPLER - sampler_budget_2x,
}

# ---- the sampler's own floor, built from measured terms -----------------------------------
# None of these levers removes a unit of the model's own work: a trace replays the identical
# 1066 programs, a distributed LayerNorm computes the identical normalisation on more cores,
# and a fusion computes the identical values in one kernel instead of five. 200 sampling steps,
# 3 recycles and the 35-row alignment are untouched by every one of them.
#
# Launch latency is NOT assumed to vanish under trace. The Pairformer block is the control: its
# host is fully ahead (132 us/op against ~20 us of issue), so its span-minus-kernel of 0.3476 ms
# over 272 programs is the device's own program-to-program latency, 1.278 us/program on Blackhole.
# The step's gap is 4.113 us/program; the 2.835 us/program excess is host, and that is
# 0.605 s/fold against the 0.647 s `b2z-host-residual-kill` measured independently -- 6.5 % apart,
# two unrelated instruments. Trace is credited with the host excess only; the 1.278 us/program
# device floor is carried, and an optimistic row that also hides it is reported beside every stack.
LAUNCH_US = (36.3438 - 35.9962) * 1000 / 272
N_PROGRAMS = M["census"]["programs_per_step"]
out["launch_latency"] = {
    "us_per_program_blackhole": LAUNCH_US,
    "source": "PairformerLayer span-minus-kernel, host fully ahead, b2z-kernel-cycle-census",
    "step_gap_us_per_program": gap * 1000 / N_PROGRAMS,
    "host_excess_s_per_fold_derived": (gap * 1000 / N_PROGRAMS - LAUNCH_US) * N_PROGRAMS * STEPS / 1e6,
    "host_excess_s_per_fold_measured_by_b2z_host_residual_kill": 0.647,
}

# CORRECTION to `b2z-diffusion-utilization`'s lever split, and it is the largest single finding
# of this pass. That row priced "matmul grid height on a 16-tile row axis" at 0.5042 s/fold, the
# biggest piece of its 1.068 s of idle core-seconds. It is worth essentially nothing.
# ttnn's 2D matmul costs ceil(M/gy) * ceil(N/gx) * K tile-blocks per core. The step's dominant
# matmul is M = 512 rows = 16 tiles, N = 768 = 24 tiles; on an 11x10 grid gy <= 10 and gx <= 11,
# so ceil(16/gy) = 2 for every gy in 8..10 and ceil(24/gx) = 3 for every gx in 8..11. Every grid
# from 8x8 = 64 cores to 10x11 = 110 cores does 6 tile-blocks per core. Re-spreading buys zero.
# Checked over all 387 matmuls in the step (`step_map.grid_optimality`): 0.31 % of matmul time
# sits on a sub-optimal grid, one call of 387. ttnn's 64-core choice is correct, not a defect.
MATMUL_RECOVERABLE_MS = 0.0269
LN = tbl["LayerNormDeviceOperation"]
HEADS = tbl["NlpCreateHeadsDeviceOperation"]
idle_by_op = M["occupancy"]["by_op_top"]
grid_recoverable = (idle_by_op["LayerNormDeviceOperation"]
                    + idle_by_op["NlpCreateHeadsDeviceOperation"]) * RECOVERABLE + MATMUL_RECOVERABLE_MS
out["grid_lever_corrected"] = {
    "wave1_claim_s_per_fold": 0.5042,
    "matmul_recoverable_ms_per_step": MATMUL_RECOVERABLE_MS,
    "matmul_recoverable_s_per_fold": MATMUL_RECOVERABLE_MS * STEPS / 1000,
    "layernorm_idle_ms_per_step": idle_by_op["LayerNormDeviceOperation"],
    "nlpcreateheads_idle_ms_per_step": idle_by_op["NlpCreateHeadsDeviceOperation"],
    "total_recoverable_ms_per_step": grid_recoverable,
    "total_recoverable_s_per_fold": grid_recoverable * STEPS / 1000,
    "wave1_overprice_factor": 0.5042 / (MATMUL_RECOVERABLE_MS * STEPS / 1000),
}

# post-grid cost of each op class, so the fusion levers are not credited with time the grid
# lever already removed
def respread(op):
    t = tbl[op]
    return t["ms_per_step"] * (1 - RECOVERABLE * (1 - t["occupancy_pct"] / 100.0))

plumb_after = respread("NlpCreateHeadsDeviceOperation") + sum(
    tbl[o]["ms_per_step"] for o in PLUMBING if o in tbl and o != "NlpCreateHeadsDeviceOperation")
binary_ms = tbl["BinaryNgDeviceOperation"]["ms_per_step"]
n_plumb = sum(tbl[o]["calls_per_step"] for o in PLUMBING if o in tbl)
n_binary = tbl["BinaryNgDeviceOperation"]["calls_per_step"]

k1 = kernel
k2 = k1 - grid_recoverable
k3 = k2 - plumb_after
k4 = k3 - binary_ms / 2
STACK = [
    ("today, untraced", k1, N_PROGRAMS, False),
    ("+ trace (already built, --diffusion_trace, default off)", k1, N_PROGRAMS, True),
    ("+ distributed LayerNorm (+ the one bad matmul grid)", k2, N_PROGRAMS, True),
    ("+ fuse the SDPA head plumbing", k3, N_PROGRAMS - n_plumb, True),
    ("+ fuse half of BinaryNg into its producers", k4, N_PROGRAMS - n_plumb - n_binary // 2, True),
]


def fold_of(step_ms):
    sec = step_ms * STEPS / 1000
    fold = TRUNK_FLOOR + sec + REST
    return {"step_ms": step_ms, "sampler_s": sec, "sampler_ratio": SAMPLER / sec,
            "fold_s_with_trunk_at_floor": fold, "fold_ratio_vs_cell": CELL / fold,
            "short_of_2x_s": fold - TARGET}


out["sampler_floor"] = {}
out["sampler_floor_zero_launch"] = {}
for name, km, npg, tr in STACK:
    per_prog = LAUNCH_US if tr else gap * 1000 / N_PROGRAMS
    d = fold_of(km + per_prog * npg / 1000)
    d.update({"kernel_ms": km, "programs": npg, "traced": tr})
    out["sampler_floor"][name] = d
    out["sampler_floor_zero_launch"][name] = fold_of(km)

# ---- lever table, each priced as a fraction of the step -----------------------------------
# `build_effort` is a relative integer, not an estimate in days: 1 = flip an existing default and
# measure it, 5 = swap one op for a variant tt-metal already ships, 8 = write and validate a fused
# kernel. Ranked by s/fold per unit of that.
def lever(name, ms_per_step, effort, falsifier, note):
    sec = ms_per_step * STEPS / 1000
    return {"lever": name, "ms_per_step": ms_per_step, "s_per_fold": sec,
            "pct_of_step_wall": 100 * ms_per_step / STEP_WALL_MS,
            "fold_ratio_alone_vs_cell": CELL / (CELL - sec),
            "build_effort": effort, "s_per_unit_effort": sec / effort,
            "falsifier": falsifier, "note": note}

host_excess_ms = gap - LAUNCH_US * N_PROGRAMS / 1000
out["levers"] = sorted([
    lever("trace the 200-step loop (--diffusion_trace, ALREADY BUILT, default False)",
          host_excess_ms, 1,
          "traced step wall > 24 ms => the 3.02 ms/step is not host dispatch and this row "
          "mis-attributed it; the gap would then be device-side and trace cannot touch it",
          "score_model.forward_traced, boltz2.py:4160. Measure it, do not build it. Two "
          "independent instruments agree it is worth 0.605-0.647 s/fold."),
    lever("distributed / width-sharded LayerNorm (114 calls/step, 100 of them on 16 of 110 cores)",
          idle_by_op["LayerNormDeviceOperation"] * RECOVERABLE, 5,
          "a width-sharded LayerNorm at [512,768] does not beat the 16-core row-parallel one "
          "=> it is bandwidth-bound, not core-bound, and the 14.5 %-of-grid figure is not a lever",
          "the programs log DistributedLayerNormStage::NOT_DISTRIBUTED; tt-metal already ships "
          "the distributed variant."),
    lever("fuse the SDPA head plumbing (NlpCreateHeads/Reshape/Slice/Transpose/Concat, 153 programs)",
          plumb_after, 8,
          "a fused head path does not beat the 5-op chain => the plumbing is bandwidth-bound and "
          "only a layout change upstream, not a fusion, can touch it",
          "costs %.2fx the SDPA it feeds, and deletes 153 of 1066 programs, so it takes launch "
          "latency with it." % (plumb / tbl["SDPAOperation"]["ms_per_step"])),
    lever("fuse BinaryNg into its producers (339 calls/step: AdaLN modulation + residual adds)",
          binary_ms / 2, 8,
          "DST-resident fusion of an eltwise into its producer returns <50 % of the eltwise => "
          "the trunk's 'BinaryNg is the cleanest delete in the block' finding does not transfer",
          "half credited; b2z2-dst-resident-fusion owns the mechanism. 339 ops at 10.6 us."),
    lever("re-spread the diffusion matmuls onto more cores  [REFUTED, priced for the record]",
          MATMUL_RECOVERABLE_MS, 5,
          "n/a -- already refuted: 386 of 387 matmuls are on their optimal grid",
          "wave 1 priced this at 0.5042 s/fold. It is worth 0.0054 s/fold."),
], key=lambda d: -d["s_per_unit_effort"])

(HERE / "ceiling.json").write_text(json.dumps(out, indent=1))

# ---- report --------------------------------------------------------------------------------
p = print
p("=== ONE DIFFUSION STEP, 512 aa, qb2 card 0 Blackhole, 11x10 grid ===")
p(f"  wall          {STEP_WALL_MS:7.3f} ms   MEASURED (perf page cell 5.280 s / 200, fc7fed56)")
p(f"  kernel        {kernel:7.3f} ms   MEASURED (1066 programs, b2z-kernel-cycle-census)")
p(f"  dispatch gap  {gap:7.3f} ms   DERIVED  wall - kernel  ({100*gap/STEP_WALL_MS:.1f} % of wall)")
p(f"  grid underfill{idle:7.3f} ms   MEASURED ({100*idle/kernel:.1f} % of kernel, {RECOVERABLE*100:.1f} % of it above a roof)")
p(f"  zero-math ops {zm_ms:7.3f} ms   MEASURED ({100*zm_ms/kernel:.1f} % of kernel does no arithmetic)")
p(f"  SDPA          {tbl['SDPAOperation']['ms_per_step']:7.3f} ms   vs {plumb:.3f} ms of plumbing to feed it "
  f"({plumb/tbl['SDPAOperation']['ms_per_step']:.2f}x)")
p()
p("=== WHAT THE SAMPLER MUST DELIVER (trunk at its 4.689 s movement-free floor, rest held) ===")
r = out["required"]
p(f"  2x = {TARGET:.3f} s.  trunk floor {TRUNK_FLOOR:.3f} + rest {REST:.3f} = {TRUNK_FLOOR+REST:.3f} s")
p(f"  => sampler budget {r['sampler_s_for_2x_with_trunk_at_floor']:.3f} s "
  f"= {r['sampler_ms_per_step_required']:.2f} ms/step, a {r['sampler_ratio_required']:.3f}x on the sampler")
p()
p("=== THE SAMPLER'S OWN FLOOR (trunk held at its 4.689 s movement-free floor, rest at measured) ===")
p(f"{'regime':56s} {'progs':>6s} {'ms/step':>8s} {'sampler':>8s} {'fold':>7s} {'vs cell':>9s} {'2x gap':>8s}")
for k, v in out["sampler_floor"].items():
    z = out["sampler_floor_zero_launch"][k]
    p(f"{k:56s} {v['programs']:6d} {v['step_ms']:8.3f} {v['sampler_s']:8.3f} {v['fold_s_with_trunk_at_floor']:7.3f} "
      f"{v['fold_ratio_vs_cell']:8.4f}x {v['short_of_2x_s']:+8.3f}   [zero-launch {z['fold_ratio_vs_cell']:.4f}x]")
p()
g = out["grid_lever_corrected"]
p(f"  CORRECTION: wave 1 priced the matmul grid lever at {g['wave1_claim_s_per_fold']:.4f} s/fold. "
  f"Measured recoverable: {g['matmul_recoverable_s_per_fold']:.4f} s/fold "
  f"({g['wave1_overprice_factor']:.0f}x overpriced).")
p()
p("=== LEVERS, ranked by s/fold per unit build effort ===")
for l in out["levers"]:
    p(f"  {l['s_per_fold']:5.3f} s/fold  {l['pct_of_step_wall']:5.1f} % of step  "
      f"effort {l['build_effort']}  -> {l['s_per_unit_effort']:.3f}  | {l['lever'][:70]}")
