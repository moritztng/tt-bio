#!/usr/bin/env python3
"""What binds the 512 aa fold's committed floor, by op class, and why that cannot be read at burst.

The roof campaign's floor of record takes max(traffic, arithmetic) per op. This splits its 12.706 s
by class and by which term binds, and locates the clock provenance of the roof it is computed
against. CPU only; opens no device and measures nothing new.
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
    by, total = c["by_op"], c["total_floor_s"]
    rows = sorted(
        ({"op": k, "calls": v["calls"], "floor_s": v["s_floor"],
          "traffic_s": v["s_traffic"], "arith_s": v["s_arith"],
          "binds": "arithmetic" if v["s_arith"] >= v["s_traffic"] else "traffic"}
         for k, v in by.items() if isinstance(v, dict)),
        key=lambda x: -x["floor_s"])
    arith = sum(x["floor_s"] for x in rows if x["binds"] == "arithmetic")
    traffic = total - arith
    top3 = rows[:3]
    out = {
        "scope": "CPU split of a committed floor artifact. No device, no new timing.",
        "census": str(CENSUS),
        "total_calls": c["total_calls"],
        "total_floor_s": total,
        "by_binding_term": {
            "arithmetic_s": arith, "arithmetic_pct": 100 * arith / total,
            "traffic_s": traffic, "traffic_pct": 100 * traffic / total,
        },
        "top_classes": [dict(x, pct_of_floor=100 * x["floor_s"] / total) for x in rows[:8]],
        "concentration": {
            "classes": [x["op"] for x in top3],
            "calls": sum(x["calls"] for x in top3),
            "floor_s": sum(x["floor_s"] for x in top3),
            "pct_of_floor": 100 * sum(x["floor_s"] for x in top3) / total,
            "pct_of_calls": 100 * sum(x["calls"] for x in top3) / c["total_calls"],
        },
        "roof_provenance": {
            "file": str(ROOFS),
            "host": r.get("host"), "card": r.get("card"), "grid": r.get("grid"),
            "cube4096_TFLOPs": r.get("cube4096_TFLOPs"),
            "loadavg_at_measurement": r.get("loadavg"),
            "recorded_clock": None,
            "problem": "The compute roof every floor second is divided by carries NO clock and was "
                       "taken at loadavg 6.2. An arithmetic-bound floor second scales with AICLK, "
                       "so an unclocked roof makes every arithmetic-bound floor second unreadable, "
                       "and with it the 74/26 split below.",
        },
        "why_the_mix_moves_at_burst": [
            "Arithmetic-bound time scales with AICLK: the same FLOPs take fewer seconds at 1350 MHz.",
            "DRAM-bandwidth-bound time does not: DRAM runs on its own clock.",
            "L1 and NOC bandwidth DO track the core clock, so a traffic term is only AICLK-immune "
            "to the extent it is DRAM rather than L1/NOC. This census does not separate them, "
            "which is one more reason the split has to be re-measured rather than rescaled.",
            "So at burst clock the arithmetic side shrinks, the DRAM side does not, and the "
            "binding mix moves toward traffic by an amount this artifact cannot tell you.",
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
