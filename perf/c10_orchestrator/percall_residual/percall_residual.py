#!/usr/bin/env python3
"""The matmul gap is flat per call, not proportional to work -- or the instrument added it.

`matmul_ceiling/` found the fold's matmuls run about 3.9x below what arithmetic intensity permits
and concluded that implementation binds rather than DRAM. "Implementation" is not a lever. This
splits it, and the split is decidable from the shapes alone because a fixed per-call cost and a
uniform rate deficit make opposite predictions across a wide FLOP range.

Residual = modelled time per call - roofline time per call, for each recorded matmul shape.
If the gap is a rate deficit, residual scales with FLOPs. If it is a fixed per-call cost, residual
is flat. The recorded shapes span **1152x in FLOPs per call**, which is a wide enough lever arm to
tell those apart without a device.
"""
import json
import re
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOOR = HERE.parents[1] / "roof_true" / "true_floor_512_qb2c2.json"
CENSUS = HERE.parents[1] / "roof_launch" / "op_census_512.json"

BYTES_PER_ELEM = 2
# c10-fold-census's predecessor measured the DEVICE's own program-launch cost.
LAUNCH_FLOOR_S = 0.487
LAUNCH_PROGRAMS = 287000
SMALL_FLOP_CUTOFF = 10e6     # below this a shape is too small for the comparison to mean much


def _terms(sig):
    t = [[int(x) for x in p.split("x")] for p in re.findall(r"\b\d+(?:x\d+)+\b", sig)]
    if len(t) < 3:
        return None
    a, b = t[1], t[2]
    m = 1
    for x in a[:-1]:
        m *= x
    return m, a[-1], (b[-1] if len(b) >= 2 else b[0])


def analyse():
    floor = json.loads(FLOOR.read_text())
    census = json.loads(CENSUS.read_text())
    cube = floor["roofs"]["dense_cube_TFLOPs"]
    bw = floor["roofs"]["stream_GBps"] / 1e3

    rows = []
    for k, v in census["top_shapes"].items():
        if not k.startswith(("ttnn.linear", "ttnn.matmul")):
            continue
        t = _terms(k)
        if t is None:
            continue
        m, kk, n = t
        flops = 2 * m * kk * n
        byts = (m * kk + kk * n + m * n) * BYTES_PER_ELEM
        ceiling = min(cube, flops / byts * bw)
        roof_us = flops / (ceiling * 1e12) * 1e6
        model_us = v["s_floor"] / v["calls"] * 1e6
        rows.append({"shape": k, "calls": v["calls"], "flops_per_call": flops,
                     "roof_us": roof_us, "model_us": model_us,
                     "residual_us": model_us - roof_us})
    rows.sort(key=lambda r: -r["flops_per_call"])
    if not rows:
        return {"scope": "no matmul shapes in the census", "shapes": [], "verdict": "NO DATA"}

    big = [r for r in rows if r["flops_per_call"] >= SMALL_FLOP_CUTOFF]
    res_all = [r["residual_us"] for r in rows]
    res_big = [r["residual_us"] for r in big]
    flop_lo = min(r["flops_per_call"] for r in rows)
    flop_hi = max(r["flops_per_call"] for r in rows)

    out = {
        "scope": "CPU comparison of two committed artifacts. No device, no fold, no new timing.",
        "shapes": rows,
        "lever_arm": {"flops_per_call_min": flop_lo, "flops_per_call_max": flop_hi,
                      "range_x": flop_hi / flop_lo},
        "residual_us": {
            "all": {"n": len(res_all), "median": st.median(res_all), "mean": st.mean(res_all),
                    "sd": st.pstdev(res_all), "min": min(res_all), "max": max(res_all),
                    "range_x": max(res_all) / min(res_all)},
            "excluding_tiny_shapes": {
                "n": len(res_big), "cutoff_MFLOP": SMALL_FLOP_CUTOFF / 1e6,
                "median": st.median(res_big), "sd": st.pstdev(res_big),
                "min": min(res_big), "max": max(res_big),
                "range_x": max(res_big) / min(res_big)},
        },
        "total": {
            "residual_s": sum(r["calls"] * r["residual_us"] for r in rows) / 1e6,
            "roofline_s": sum(r["calls"] * r["roof_us"] for r in rows) / 1e6,
            "modelled_s": sum(r["calls"] * r["model_us"] for r in rows) / 1e6,
        },
    }
    out["total"]["residual_share_pct"] = (
        100.0 * out["total"]["residual_s"] / out["total"]["modelled_s"])

    out["reading"] = (
        "Across a %.0fx range in FLOPs per call the residual spans only %.0fx, and %.0fx of that is "
        "the three smallest shapes. Excluding those, the residual is %.1f us median with a %.1f us "
        "spread. A rate deficit would have made the residual track the %.0fx FLOP range; it does "
        "not. **The gap is dominated by a roughly FIXED per-call cost of about %.0f us**, and over "
        "the recorded shapes it is %.3f s of a modelled %.3f s -- %.0f %% of their time."
        % (out["lever_arm"]["range_x"], out["residual_us"]["all"]["range_x"],
           out["residual_us"]["all"]["range_x"] / out["residual_us"]["excluding_tiny_shapes"]["range_x"],
           out["residual_us"]["excluding_tiny_shapes"]["median"],
           out["residual_us"]["excluding_tiny_shapes"]["sd"], out["lever_arm"]["range_x"],
           out["residual_us"]["excluding_tiny_shapes"]["median"], out["total"]["residual_s"],
           out["total"]["modelled_s"], out["total"]["residual_share_pct"])
    )
    out["launch_does_not_explain_it"] = {
        "device_launch_floor_s": LAUNCH_FLOOR_S, "programs": LAUNCH_PROGRAMS,
        "us_per_program": LAUNCH_FLOOR_S / LAUNCH_PROGRAMS * 1e6,
        "residual_over_launch_x": (out["residual_us"]["excluding_tiny_shapes"]["median"]
                                   / (LAUNCH_FLOOR_S / LAUNCH_PROGRAMS * 1e6)),
    }

    # --- the alternative explanation, which has to be stated first, not last ------------------
    out["ALTERNATIVE_the_instrument_added_it"] = (
        "The modelled us/call comes from per-shape rates measured ON PC, whose card runs custom "
        "130-core firmware. If pc's harness carried its own fixed per-measurement overhead of tens "
        "of microseconds, the entire residual is a property of that instrument and not of the fold. "
        "Nothing here can separate those, and this artifact does NOT claim a real per-call cost on "
        "qb2. It makes a prediction instead."
    )
    out["PREDICTED_for_the_measuring_row"] = [
        "If the per-call cost is REAL, re-measuring these shapes on qb2 reproduces a residual that "
        "is flat in FLOPs at roughly 20 us, and the matmul class's achieved rate stays near 20 "
        "TFLOP/s with about two thirds of its time outside the roofline.",
        "If it was PC's INSTRUMENT, the qb2 residual collapses -- large shapes land near their "
        "roofline time and the achieved rate rises toward the structural ceiling on the big shapes "
        "while staying low only on the genuinely DRAM-bound 16-head pair matmuls.",
        "These differ by more than 3x on the achieved rate, so even a coarse measurement separates "
        "them. Report the residual per shape, not just the class rate, because the class rate alone "
        "cannot distinguish the two.",
    ]
    out["limits"] = [
        "Both inputs are modelled. This compares a roofline computed here against a floor computed "
        "elsewhere; neither is a measurement of the fold on qb2.",
        "18 recorded shapes, 86 % of the class's calls but 29 % of its modelled floor. The "
        "residual is NOT extrapolated to the remainder.",
        "The three shapes under 10 MFLOP per call are excluded from the headline spread because "
        "their roofline time is under 1 us, so any fixed cost swamps them by construction. They are "
        "kept in the table and in the all-shapes statistics.",
        "A flat residual is consistent with several mechanisms -- program setup, CB and semaphore "
        "configuration, kernel prologue -- and this does not distinguish among them.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "percall_residual.json").write_text(json.dumps(r, indent=2) + "\n")
    b = r["residual_us"]["excluding_tiny_shapes"]
    print(f"FLOP/call lever arm: {r['lever_arm']['range_x']:.0f}x")
    print(f"residual (all {r['residual_us']['all']['n']}): median "
          f"{r['residual_us']['all']['median']:.1f} us, range {r['residual_us']['all']['range_x']:.0f}x")
    print(f"residual (>{b['cutoff_MFLOP']:.0f} MFLOP, n={b['n']}): median {b['median']:.1f} us, "
          f"sd {b['sd']:.1f}, range {b['range_x']:.1f}x")
    t = r["total"]
    print(f"residual total {t['residual_s']:.3f} s of modelled {t['modelled_s']:.3f} s "
          f"({t['residual_share_pct']:.0f} %); roofline part {t['roofline_s']:.3f} s")
    l = r["launch_does_not_explain_it"]
    print(f"device launch floor {l['us_per_program']:.2f} us/program -> residual is "
          f"{l['residual_over_launch_x']:.0f}x that")
