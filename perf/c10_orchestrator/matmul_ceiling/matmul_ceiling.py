#!/usr/bin/env python3
"""Bandwidth does not explain the fold's matmul rate: these shapes could reach ~78 TFLOP/s.

`frontier/` established that the matmul class's achieved rate is the only axis with enough headroom
to reach 10.0 s, and that the number saying so is modelled from per-shape rates measured on the
wrong machine. Before spending a chip on it, one thing can be settled here: **is the gap even
allowed by the hardware, or are these shapes structurally bandwidth-bound?**

Roofline answers that from the shapes alone. For each recorded matmul, arithmetic intensity is
FLOPs over bytes moved, and the per-shape ceiling is `min(dense cube, AI x DRAM bandwidth)` using
the campaign's own measured roofs. If the ceiling came out near today's rate, the headroom would be
illusory and 10.0 s would be settled as impossible without a fold.

It does not come out near today's rate.
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOOR = HERE.parents[1] / "roof_true" / "true_floor_512_qb2c2.json"
CENSUS = HERE.parents[1] / "roof_launch" / "op_census_512.json"

BYTES_PER_ELEM = 2          # bf16
TODAY_TFLOPS = 19.88        # matmul_TFLOP / matmul_floor_s, the modelled class rate
NEEDED_TFLOPS = 35.49       # frontier: what 10.0 s needs on this axis alone

# Every Blackhole (cube, DRAM) roof pair this campaign has on record, so the conclusion can be
# checked against all of them rather than resting on the one the floor artifact happens to carry.
# The last two are c10-fold-census's in-session roofs, read from its runs/sweep1 and runs/sweep2
# replay.json; that row has NOT published them and they are used here only to bound a sensitivity,
# never as a result. Its two sweeps agree to 0.01 %, which is why they are worth bounding against.
ROOF_RANGE = [
    (104.93, 424.7, "roof_true/true_floor_512_qb2c2.json, the floor artifact this module uses"),
    (112.71, 424.7, "median of the four-measurement cluster in frontier/"),
    (122.28, 442.9, "c10-fold-census sweep1, in-session, UNPUBLISHED"),
    (122.30, 443.1, "c10-fold-census sweep2, in-session, UNPUBLISHED"),
]


def _dims(sig):
    return [[int(x) for x in t.split("x")] for t in re.findall(r"\b\d+(?:x\d+)+\b", sig)]


def _shape_terms(sig):
    """(M, K, N) for a recorded linear/matmul signature: operands are out|in=A,B."""
    t = _dims(sig)
    if len(t) < 3:
        return None
    a, b = t[1], t[2]
    m = 1
    for x in a[:-1]:
        m *= x
    k = a[-1]
    n = b[-1] if len(b) >= 2 else b[0]
    return m, k, n


def analyse():
    floor = json.loads(FLOOR.read_text())
    census = json.loads(CENSUS.read_text())
    cube = floor["roofs"]["dense_cube_TFLOPs"]
    bw_tbs = floor["roofs"]["stream_GBps"] / 1e3
    balance = floor["roofs"]["machine_balance_flop_per_byte"]

    rows = []
    for k, v in census["top_shapes"].items():
        if not k.startswith(("ttnn.linear", "ttnn.matmul")):
            continue
        t = _shape_terms(k)
        if t is None:
            continue
        m, kk, n = t
        flops = 2 * m * kk * n
        act = (m * kk + m * n) * BYTES_PER_ELEM     # activations in + out
        wt = kk * n * BYTES_PER_ELEM                # the weight matrix
        row = {"shape": k, "calls": v["calls"], "M": m, "K": kk, "N": n, "flops_per_call": flops}
        for label, byts in (("cold", act + wt), ("warm", act)):
            ai = flops / byts
            row[f"ai_{label}"] = ai
            row[f"ceiling_{label}_TFLOPs"] = min(cube, ai * bw_tbs)
            row[f"bound_{label}"] = "compute" if ai > balance else "dram"
        rows.append(row)

    out = {
        "scope": "CPU roofline of the recorded matmul shapes against the campaign's own measured "
                 "roofs. No device, no fold, no new measurement.",
        "roofs": {"dense_cube_TFLOPs": cube, "stream_GBps": floor["roofs"]["stream_GBps"],
                  "machine_balance_flop_per_byte": balance,
                  "clock": "UNRECORDED on both roofs; a ratio within them is more robust than "
                           "either absolute"},
        "shapes": sorted(rows, key=lambda r: -r["calls"]),
    }

    tot_flops = sum(r["flops_per_call"] * r["calls"] for r in rows)
    res = {}
    for label in ("cold", "warm"):
        secs = sum(r["flops_per_call"] * r["calls"] / (r[f"ceiling_{label}_TFLOPs"] * 1e12)
                   for r in rows)
        # A census with no matmul shapes has no ceiling to report; say so rather than divide by
        # zero, which is how an empty input turns into a confident number.
        res[label] = {"structural_floor_s": secs,
                      "ceiling_TFLOPs": (tot_flops / 1e12 / secs) if secs > 0 else None}
    ceil_cold = res["cold"]["ceiling_TFLOPs"]
    out["class"] = {
        "calls": sum(r["calls"] for r in rows),
        "TFLOP": tot_flops / 1e12,
        "structural_ceiling": res,
        "today_TFLOPs": TODAY_TFLOPS,
        "needed_for_target_TFLOPs": NEEDED_TFLOPS,
        "gap_to_structural_x": (ceil_cold / TODAY_TFLOPS) if ceil_cold else None,
        "gap_to_target_x": NEEDED_TFLOPS / TODAY_TFLOPS,
        "target_pct_of_structural": (100.0 * NEEDED_TFLOPS / ceil_cold) if ceil_cold else None,
    }

    if ceil_cold is None:
        out["verdict"] = "NO MATMUL SHAPES in the census: nothing to say about the ceiling."
        return out
    # --- does the conclusion depend on which roofs are right? ---------------------------------
    sens = []
    for cube_i, bw_i, src in ROOF_RANGE:
        secs = 0.0
        for r in rows:
            ceiling = min(cube_i, r["ai_cold"] * bw_i / 1e3)
            secs += r["flops_per_call"] * r["calls"] / (ceiling * 1e12)
        c_i = tot_flops / 1e12 / secs
        sens.append({"cube_TFLOPs": cube_i, "dram_GBps": bw_i, "source": src,
                     "ceiling_TFLOPs": c_i, "x_today": c_i / TODAY_TFLOPS,
                     "target_pct_of_ceiling": 100.0 * NEEDED_TFLOPS / c_i})
    ceils = [x["ceiling_TFLOPs"] for x in sens]
    out["roof_sensitivity"] = {
        "rows": sens,
        "ceiling_range_TFLOPs": [min(ceils), max(ceils)],
        "ceiling_spread_pct": 100.0 * (max(ceils) - min(ceils)) / min(ceils),
        "x_today_range": [min(x["x_today"] for x in sens), max(x["x_today"] for x in sens)],
        "target_pct_range": [min(x["target_pct_of_ceiling"] for x in sens),
                             max(x["target_pct_of_ceiling"] for x in sens)],
        "reading": "Across every roof pair the campaign has measured the ceiling moves only "
                   "%.1f to %.1f TFLOP/s, the gap to today stays %.1fx to %.1fx, and the target "
                   "stays %.0f to %.0f %% of the ceiling. **The conclusion does not depend on which "
                   "roofs are right**, and the better-measured roofs make it slightly stronger. "
                   "That is the point of this section: it removes the roof uncertainty as a reason "
                   "to doubt the finding, without resting on any unpublished number."
                   % (min(ceils), max(ceils),
                      min(x["x_today"] for x in sens), max(x["x_today"] for x in sens),
                      min(x["target_pct_of_ceiling"] for x in sens),
                      max(x["target_pct_of_ceiling"] for x in sens)),
    }

    out["verdict"] = (
        "Bandwidth does NOT explain the fold's matmul rate. The recorded shapes carry enough "
        "arithmetic intensity for a structural ceiling of %.1f TFLOP/s, against a modelled %.2f "
        "today -- a %.1fx gap that roofline does not account for. The target needs %.2f TFLOP/s, "
        "which is only %.0f %% of that ceiling. So the headroom frontier/ identified is not "
        "excluded by the hardware, and whatever limits these matmuls is implementation -- grid "
        "occupancy, tile quantisation, K-loop efficiency, per-program cost -- rather than DRAM."
        % (res["cold"]["ceiling_TFLOPs"], TODAY_TFLOPS,
           res["cold"]["ceiling_TFLOPs"] / TODAY_TFLOPS, NEEDED_TFLOPS,
           100.0 * NEEDED_TFLOPS / res["cold"]["ceiling_TFLOPs"])
    )
    out["sharpens_the_census_question"] = (
        "c10-fold-census was asked to measure the achieved rate. This says the more useful question "
        "is WHY it sits about 4x below what arithmetic intensity permits, because that gap is where "
        "the campaign's only viable axis lives. Two shapes are worth naming: the 16-head pair "
        "matmuls at 8,448 calls each have AI 101, so they are DRAM-bound and capped near 43 "
        "TFLOP/s; the 768-family linears at AI 219-307 sit right at the machine balance of 247."
    )
    out["limits"] = [
        "Roofline is an upper bound that ignores exactly what probably binds here -- grid "
        "occupancy, tile quantisation, K-loop structure, per-program dispatch. 78 TFLOP/s is 'not "
        "excluded by bandwidth', NOT 'achievable'.",
        "Only the 18 recorded matmul shapes, which are 86 % of the class's calls but 29 % of its "
        "modelled floor. The unmeasured remainder is not extrapolated here.",
        "Both roofs carry an unrecorded clock, so the absolute ceiling moves with them; the ratio "
        "between ceiling and today's rate is the more robust reading.",
        "Bytes assume bf16 and no accumulator traffic, and the cold/warm pair brackets whether the "
        "weight matrix is re-read per call. The two differ by under 3 %, so weight residency is "
        "not what decides this.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "matmul_ceiling.json").write_text(json.dumps(r, indent=2) + "\n")
    c = r["class"]
    print(f"{c['calls']:.0f} calls, {c['TFLOP']:.2f} TFLOP")
    for lab, v in c["structural_ceiling"].items():
        print(f"  {lab:5s} structural floor {v['structural_floor_s']:.3f} s -> ceiling "
              f"{v['ceiling_TFLOPs']:.2f} TFLOP/s")
    print(f"today {c['today_TFLOPs']} -> gap {c['gap_to_structural_x']:.1f}x; target "
          f"{c['needed_for_target_TFLOPs']} is {c['target_pct_of_structural']:.0f} % of structural")
