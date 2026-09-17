#!/usr/bin/env python3
"""Half the fold's bytes move through ops that do none of its arithmetic.

`c10-trace-lever` killed the ladder's largest item and `floor_vs_measured/` showed the clock-immune
term F = 3.9830 s is not host overhead. That leaves bytes as the only known lever against 27 % of
the fold, and the corpus's byte work was aimed at the wrong place: bfp8 and matmul fusion attack
the compute ops. This asks where the bytes actually are.

The core number is ROOF-FREE. Call counts and byte counts are graph facts, and the campaign's two
independent in-house instruments agree on the fold call count exactly and on its byte count to 3 parts
per million. Only the conversion of bytes into seconds needs a roof, and the two instruments disagree by 22 % there, so the cost is
reported as a bracket rather than a point.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CENSUS = HERE.parents[1] / "roof_launch" / "op_census_512.json"
FLOOR = HERE.parents[1] / "roof_true" / "true_floor_512_qb2c2.json"

F_MEASURED_S = 3.9830        # c10-fixed-cost c4e6725bd, clock-immune term at 512 aa
F_SE_S = 0.1181
# perf/c10_orchestrator/shape_rank/: this project has measured that folding an elementwise epilogue
# into its producing matmul returns about a third of the traffic it deletes, not all of it.
FUSION_RETURN = 1.0 / 3.0
TILE_B = 32 * 32 * 2         # a bf16 tile, which is the unit the census counts in
# Shapes whose recorded zero bytes are SEMANTICALLY correct rather than a counting hole: an
# allocation moves nothing, and a metadata-only reshape or unsqueeze moves nothing on device.
DEFENSIBLE_ZERO = ("allocate_tensor_on_device", "reshape", "unsqueeze", "squeeze")


def load():
    return json.loads(CENSUS.read_text()), json.loads(FLOOR.read_text())


def analyse():
    census, floor = load()
    by = census["by_op"]
    tot_calls = sum(v["calls"] for v in by.values())
    tot_B = sum(v["B"] for v in by.values())

    # An op the model prices at zero arithmetic. Split it, because "does no arithmetic" covers two
    # very different things and conflating them is how a lever gets aimed at bookkeeping.
    zero = {k: v for k, v in by.items() if v["s_arith"] == 0}
    moving = {k: v for k, v in zero.items() if v["B"] > 0}
    booking = {k: v for k, v in zero.items() if v["B"] == 0}

    def agg(d):
        return {"classes": len(d), "calls": sum(v["calls"] for v in d.values()),
                "TB": sum(v["B"] for v in d.values()) / 1e12,
                "s_traffic": sum(v["s_traffic"] for v in d.values())}

    out = {
        "scope": "CPU re-reading of two committed capture artifacts. No device, no new measurement.",
        "sources": {"census": str(CENSUS), "floor": str(FLOOR)},
        "fold_totals": {"calls": tot_calls, "TB": tot_B / 1e12,
                        "matmul_TFLOP": floor["matmul_TFLOP"],
                        "nonmatmul_TFLOP": floor["nonmatmul_TFLOP"]},
    }
    out["fold_totals"]["nonmatmul_pct_of_FLOP"] = 100.0 * floor["nonmatmul_TFLOP"] / (
        floor["matmul_TFLOP"] + floor["nonmatmul_TFLOP"])

    m, b = agg(moving), agg(booking)
    out["zero_arithmetic"] = {
        "byte_moving": dict(m, pct_of_calls=100.0 * m["calls"] / tot_calls,
                            pct_of_bytes=100.0 * m["TB"] * 1e12 / tot_B),
        "bookkeeping_zero_byte": dict(b, pct_of_calls=100.0 * b["calls"] / tot_calls,
                                      note="deallocate, getitem, allocate, unsqueeze, squeeze. "
                                           "Priced at exactly zero by both instruments and moving "
                                           "no device bytes. c10-trace-lever's null says host "
                                           "per-call cost is not this fold's constraint, so these "
                                           "are probably genuinely cheap -- but they are 32 % of "
                                           "the fold's calls priced at zero, which is an "
                                           "assumption, not a measurement."),
    }

    # --- the two instruments, side by side ----------------------------------------------------
    fb = floor["buckets"]
    fz_calls = sum(v["n"] for v in fb.values() if v["s_ar"] == 0)
    fz_TB = sum(v["B"] for v in fb.values() if v["s_ar"] == 0) / 1e12
    fz_s = sum(v["s_tr"] for v in fb.values() if v["s_ar"] == 0)
    zero_all = agg(zero)
    out["instrument_agreement"] = {
        "fold_calls": {"census": tot_calls, "floor": sum(v["n"] for v in fb.values())},
        "fold_TB": {"census": tot_B / 1e12, "floor": sum(v["B"] for v in fb.values()) / 1e12},
        "zero_arith_calls": {"census": zero_all["calls"], "floor": fz_calls},
        "zero_arith_TB": {"census": zero_all["TB"], "floor": fz_TB},
        "zero_arith_seconds": {"census": zero_all["s_traffic"], "floor": fz_s},
        "reading": "The graph facts agree and the roof-derived seconds do not. Call totals "
                   "match exactly, byte totals to 3 parts per million, and zero-arithmetic call "
                   "counts to 0.5 %, while the byte "
                   "attribution differs by 22 % and the seconds by 22 %. So the composition claim "
                   "is safe and any single-number cost is not.",
    }

    lo, hi = sorted([zero_all["s_traffic"], fz_s])
    out["cost_bracket_s"] = {
        "low": lo, "high": hi,
        "measured_F_s": F_MEASURED_S, "measured_F_se_s": F_SE_S,
        "F_inside_bracket": lo <= F_MEASURED_S <= hi,
        "reading": "Both in-house roofs bracket the measured clock-immune term. That is consistent "
                   "with F being arithmetic-free data movement, and it is not proof: two roofs "
                   "that disagree by 22 % will bracket a lot of things.",
    }

    # --- where it is concentrated -------------------------------------------------------------
    top = sorted(moving.items(), key=lambda kv: -kv[1]["B"])[:6]
    out["concentration"] = [
        {"op": k, "calls": v["calls"], "TB": v["B"] / 1e12,
         "pct_of_fold_bytes": 100.0 * v["B"] / tot_B, "s_traffic_census": v["s_traffic"]}
        for k, v in top
    ]
    big3 = [k for k in ("ttnn.multiply_", "ttnn.layer_norm", "ttnn.add_") if k in moving]
    b3_B = sum(moving[k]["B"] for k in big3)
    b3_s_lo = sum(moving[k]["s_traffic"] for k in big3)
    # Rescale those ops onto the other instrument's roof. Guard the degenerate case: a census
    # with no zero-arithmetic traffic at all has no scale factor and must not divide by zero.
    scale = (fz_s / zero_all["s_traffic"]) if zero_all["s_traffic"] > 0 else 1.0
    b3_s_hi = b3_s_lo * scale
    out["headline_candidate"] = {
        "ops": big3,
        "calls": sum(moving[k]["calls"] for k in big3),
        "TB": b3_B / 1e12,
        "pct_of_fold_bytes": 100.0 * b3_B / tot_B,
        "traffic_s_bracket": [b3_s_lo, b3_s_hi],
        "realistic_prize_s_bracket": [b3_s_lo * FUSION_RETURN, b3_s_hi * FUSION_RETURN],
        "why_not_the_full_amount": "These are in-place elementwise and normalisation ops over the "
                                   "pair and single representations -- pure DRAM round trips. The "
                                   "shape that a producer's epilogue can absorb. But this project "
                                   "has measured that class of fusion before and it returns about "
                                   "a third of what it deletes, so the prize is a third of the "
                                   "traffic, not all of it.",
    }

    # --- the census's own identity, and the seven shapes that break it ------------------------
    # For every recorded shape carrying both tiles and bytes, B == calls * tiles * TILE_B. That
    # identity is the census's own, it holds at a median of exactly 1.000, and seven shapes record
    # nonzero tiles against exactly zero bytes.
    shapes = census["top_shapes"]
    conform, holes = [], []
    for k, v in shapes.items():
        tiles = v["in_tiles"] + v["out_tiles"]
        if tiles <= 0:
            continue
        implied = v["calls"] * tiles * TILE_B
        if v["B"] > 0:
            conform.append(v["B"] / implied)
        else:
            holes.append({"shape": k, "calls": v["calls"], "tiles_per_call": tiles,
                          "implied_GB": implied / 1e9,
                          "defensible": any(d in k for d in DEFENSIBLE_ZERO)})
    conform.sort()
    med = conform[len(conform) // 2] if conform else None
    real = [h for h in holes if not h["defensible"]]
    correction_B = sum(h["implied_GB"] for h in real) * 1e9
    out["byte_counter_holes"] = {
        "identity": "B == calls * (in_tiles + out_tiles) * 2048",
        "shapes_conforming": len(conform),
        "median_ratio": med,
        "shapes_with_tiles_but_zero_bytes": holes,
        "defensible_note": "an allocation moves nothing and a metadata-only reshape or unsqueeze "
                           "moves nothing on device, so those zeros are semantics, not a hole",
        "real_holes": [h["shape"] for h in real],
        "correction_GB": correction_B / 1e9,
        "correction_pct_of_fold_bytes": 100.0 * correction_B / tot_B,
        "direction": "Both real holes are in-place elementwise or normalisation ops, so they belong "
                     "to the zero-arithmetic class. Correcting them RAISES this directory's "
                     "headline, which means the published figure is the conservative one.",
        "corrected_zero_arith_pct_of_bytes":
            100.0 * (m["TB"] * 1e12 + correction_B) / (tot_B + correction_B),
        "corrected_big3_pct_of_bytes": None,   # filled below once big3 is known
        "third_defect": "This is the third byte-counting defect this campaign has found, after "
                        "c10-roofline-reset's known-answer overcount of exactly one output tensor "
                        "(1.3333x) and the recorded dedupe-on-tensor-id error. Any row that counts "
                        "bytes should check this identity against its own capture first.",
    }

    out["what_is_new_here"] = [
        "The axis is against F, the clock-immune term, not against kernel time. It does not "
        "compete with arithmetic and it does not shrink when the clock rises, which is why it was "
        "mis-ranked while every ratio in the corpus was a wall-time ratio at an unrecorded clock.",
        "The concentration is three op classes, not a long tail: multiply_, layer_norm and add_ "
        "move 31 % of the fold's entire byte traffic and compute essentially none of it.",
        "The framing is roof-free: half the bytes do 0.05 % of the arithmetic, and that statement "
        "survives both roofs being wrong.",
    ]
    out["what_is_not_new"] = [
        "The byte axis itself is known and floor_mix/ already capped it at 1.487x of the floor, "
        "with an unknown part already consumed by shipped levers. This is a sharper target on a "
        "known axis, not a new axis.",
        "Nothing here is a lever or a measurement of one. No cycle deleted, no accuracy spent.",
    ]
    out["byte_counter_holes"]["corrected_big3_pct_of_bytes"] = 100.0 * (
        b3_B + correction_B) / (tot_B + correction_B)

    out["limits"] = [
        "'Zero arithmetic' is a MODEL statement -- an op the census prices with no arithmetic term "
        "-- not a hardware statement. A layer_norm does real SFPU work the model does not price.",
        "The two instruments disagree by 22 % on which bytes belong to this class, so the class "
        "boundary is soft even though the totals are exact.",
        "Layout traffic cannot all be deleted: tensors genuinely need different layouts, and an "
        "in-place add exists because something has to add. Sizing an axis is not finding a lever.",
        "Both artifacts were captured at an unrecorded clock; only the byte and call counts are "
        "clock-independent.",
    ]
    return out


if __name__ == "__main__":
    r = analyse()
    (HERE / "arithmetic_free_traffic.json").write_text(json.dumps(r, indent=2) + "\n")
    z = r["zero_arithmetic"]["byte_moving"]
    t = r["fold_totals"]
    print(f"zero-arithmetic byte-moving ops: {z['calls']:.0f} calls ({z['pct_of_calls']:.1f} % of "
          f"the fold), {z['TB']:.3f} TB ({z['pct_of_bytes']:.1f} % of its bytes)")
    print(f"  ... performing {t['nonmatmul_pct_of_FLOP']:.3f} % of the fold's arithmetic")
    c = r["cost_bracket_s"]
    print(f"cost bracket {c['low']:.3f} to {c['high']:.3f} s; measured F = {c['measured_F_s']:.4f} s"
          f" -> inside: {c['F_inside_bracket']}")
    h = r["headline_candidate"]
    print(f"concentrated in {', '.join(h['ops'])}: {h['TB']:.3f} TB = "
          f"{h['pct_of_fold_bytes']:.1f} % of fold bytes, traffic "
          f"{h['traffic_s_bracket'][0]:.3f}-{h['traffic_s_bracket'][1]:.3f} s, realistic prize "
          f"{h['realistic_prize_s_bracket'][0]:.3f}-{h['realistic_prize_s_bracket'][1]:.3f} s")
