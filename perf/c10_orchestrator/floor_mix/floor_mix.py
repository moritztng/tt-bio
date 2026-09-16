#!/usr/bin/env python3
"""What the 512 aa floor is made of, and what each axis could win if it went to zero.

The roof campaign's floor of record takes max(traffic, arithmetic) PER CALL, so a class is not
"bound by" one term -- some of its calls are traffic-bound and some are not, and summing a class's
two columns and comparing them says nothing. The meaningful question an aggregate can answer is a
ceiling: delete every byte and the floor becomes the arithmetic sum; make arithmetic free and it
becomes the traffic sum.

CPU only; opens no device and measures nothing new.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
CENSUS = HERE.parent.parent / "roof_launch" / "op_census_512.json"
ROOFS = HERE.parent.parent / "roof_shape" / "shape_roofs_qb2c3_shipped.json"


def main():
    c = json.loads(CENSUS.read_text())
    r = json.loads(ROOFS.read_text())
    by = {k: v for k, v in c["by_op"].items() if isinstance(v, dict)}
    A = sum(v["s_arith"] for v in by.values())
    T = sum(v["s_traffic"] for v in by.values())
    F = sum(v["s_floor"] for v in by.values())
    byte_rank = sorted(
        ({"op": k, "calls": v["calls"], "floor_s": v["s_floor"], "arith_s": v["s_arith"],
          "traffic_s": v["s_traffic"], "byte_headroom_s": v["s_floor"] - v["s_arith"]}
         for k, v in by.items()),
        key=lambda x: -x["byte_headroom_s"])
    out = {
        "scope": "CPU ceiling analysis of a committed floor artifact. No device, no new timing.",
        "census": str(CENSUS),
        "total_calls": c["total_calls"],
        "totals_s": {"floor": F, "arithmetic": A, "traffic": T,
                     "reported_total_floor": c["total_floor_s"]},
        "note_on_summing": "floor >= max(sum arithmetic, sum traffic) because max() is per call. "
                           "A class-level comparison of the two columns is not a binding verdict.",
        "ceilings": {
            "delete_every_byte": {"floor_becomes_s": A, "wins_s": F - A,
                                  "pct_of_floor": 100 * (F - A) / F, "ratio": F / A},
            "make_arithmetic_free": {"floor_becomes_s": T, "wins_s": F - T,
                                     "pct_of_floor": 100 * (F - T) / F, "ratio": F / T},
        },
        "where_the_byte_axis_lives": byte_rank[:8],
        "roof_provenance": {
            "file": str(ROOFS), "host": r.get("host"), "card": r.get("card"),
            "grid": r.get("grid"), "cube4096_TFLOPs": r.get("cube4096_TFLOPs"),
            "loadavg_at_measurement": r.get("loadavg"), "recorded_clock": None,
            "problem": "The compute roof every arithmetic second is divided by carries NO clock "
                       "and was taken at loadavg 6.2. Arithmetic seconds scale with AICLK, so "
                       "both ceilings below are only true at whatever clock that roof ran at.",
        },
        "why_the_mix_moves_at_burst": [
            "Arithmetic time scales with AICLK: the same FLOPs take fewer seconds at 1350 MHz.",
            "DRAM-bandwidth time does not: DRAM runs on its own clock.",
            "L1 and NOC bandwidth DO track the core clock, so a traffic term is AICLK-immune only "
            "to the extent it is DRAM rather than L1/NOC, and this census does not separate them.",
            "So at burst the arithmetic side shrinks, the DRAM side does not, and the byte axis is "
            "worth MORE than 1.487x there -- by an amount this artifact cannot quantify.",
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
