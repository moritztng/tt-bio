#!/usr/bin/env python3
"""The 512 aa fold's floor ranked by (op class, shape), with per-call cost.

The class-level split says where the time is; this says which SHAPE it is, which is what a lever
has to target. Per-call cost is the column that separates "many small calls" from "a few large
ones", and the two need different levers. CPU only; measures nothing.
"""
import json
import sys
from pathlib import Path

CENSUS = Path(__file__).resolve().parents[2] / "roof_launch" / "op_census_512.json"
TOP = 20


def main():
    d = json.loads(CENSUS.read_text())
    total = d["total_floor_s"]
    rows = []
    for key, v in d["top_shapes"].items():
        if v["calls"] <= 0:
            continue
        op, _, rest = key.partition("|")
        rows.append({
            "op": op, "signature": rest, "calls": v["calls"], "floor_s": v["s_floor"],
            "pct_of_floor": 100.0 * v["s_floor"] / total,
            "us_per_call": 1e6 * v["s_floor"] / v["calls"],
            "bytes": v["B"], "out_tiles": v["out_tiles"], "in_tiles": v["in_tiles"],
        })
    rows.sort(key=lambda r: -r["floor_s"])
    top = rows[:TOP]
    pure_traffic = [r for r in rows if r["op"].split(".")[-1] in
                    {"add_", "multiply_", "layer_norm", "permute", "add", "multiply"}]
    out = {
        "scope": "CPU ranking of a committed floor artifact. No device, no new timing.",
        "census": str(CENSUS),
        "total_floor_s": total,
        "total_calls": d["total_calls"],
        "distinct_shapes_recorded": len(d["top_shapes"]),
        "top": top,
        "top_share_pct": sum(r["pct_of_floor"] for r in top),
        "largest_single_item": top[0],
        "pure_traffic_elementwise": {
            "floor_s": sum(r["floor_s"] for r in pure_traffic),
            "calls": sum(r["calls"] for r in pure_traffic),
            "pct_of_floor": 100.0 * sum(r["floor_s"] for r in pure_traffic) / total,
        },
        "observations": [
            "The largest single (class, shape) in the fold is an in-place add over the PAIR "
            "representation, 1x512x512x128, at 1,416 calls and 0.453 s, 3.6 %% of the floor and "
            "319.6 us per call. It has no arithmetic term at all: it is a DRAM round trip of the "
            "pair tensor, and it is the kind of thing a producer's epilogue can absorb. Prior art "
            "says fusion of this kind returns about a third of the cost it deletes, not all of it.",
            "The 768-family linears carry a modelled 28.9 to 55.8 us per call. Against the "
            "same-session dense cube rate of 67.59 TFLOP/s those shapes are modelled far below "
            "peak, which is expected for their size but is worth confirming as MEASURED rather "
            "than modelled, because the roof behind the model carries no recorded clock.",
            "No single shape holds more than 3.6 %% of the floor. There is no giant hiding here: "
            "the fold is a long tail, and a lever has to either hit a class across many shapes or "
            "remove per-call cost rather than per-shape cost.",
        ],
        "limits": [
            "These are FLOOR seconds from max(traffic, arithmetic) per call, not measured time.",
            "The roof behind them has no recorded clock and was taken at loadavg 6.2, so the "
            "arithmetic side of every row is only valid at that unknown clock.",
            "Shapes are as the capture recorded them on an older tree.",
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
