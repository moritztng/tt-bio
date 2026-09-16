#!/usr/bin/env python3
"""Where the grid-sizing lever's evidence actually comes from, and what Blackhole has instead.

The lever ledger ranks per-class grid sizing second, on a measurement of the triangle product at
0.5932 ms on 32 cores against 1.2480 ms on 72. This reads the provenance of that measurement and
of every Blackhole arm for the same classes. CPU only; opens no device and measures nothing new.
"""
import json
import sys
from pathlib import Path

PERF = Path(__file__).resolve().parents[2]
WH = PERF / "roof_tri_arith" / "tri_close_whglx_wh.json"
BH_TRIATT = PERF / "roof_triatt_rate" / "rate_ab_512_qb2c3.json"
BH_TRIMUL = PERF / "roof_triatt_rate" / "trimul_rate_ab_512_qb2c3.json"


def main():
    wh = json.loads(WH.read_text())
    rows = {r["arm"]: r["ms"] for r in wh["rows"]}
    ta = json.loads(BH_TRIATT.read_text())
    tm = json.loads(BH_TRIMUL.read_text())

    def bh(doc, arm):
        a = doc["arms"][arm]
        return {"min_ms": a["min_ms"], "median_ms": a["median_ms"], "TFLOPs_at_min": a["TFLOPs_at_min"]}

    cube = bh(tm, "cube")
    out = {
        "scope": "CPU provenance read of committed artifacts. No device, no new timing.",
        "the_ranked_claim": "triangle product 0.5932 ms on 32 cores against 1.2480 ms on 72",
        "its_provenance": {
            "file": str(WH), "host": wh["host"], "arch": wh["arch"],
            "grid": wh["grid"], "cores": wh["cores"], "loadavg": wh["loadavg"],
            "recorded_clock": None,
            "product_72_cores_ms": rows["abl_product_dramout"],
            "product_32_cores_ms": rows["abl_product_cores32"],
            "product_16_cores_ms": rows["abl_product_cores16"],
            "product_win_32_over_72": rows["abl_product_dramout"] / rows["abl_product_cores32"],
        },
        "the_same_sweep_moves_the_other_way_for_sdpa": {
            "sdpa_72_cores_ms": rows["abl_sdpa_dramout"],
            "sdpa_32_cores_ms": rows["abl_sdpa_cores32"],
            "sdpa_16_cores_ms": rows["abl_sdpa_cores16"],
            "sdpa_ratio_32_over_72": rows["abl_sdpa_cores32"] / rows["abl_sdpa_dramout"],
            "reading": "On the SAME Wormhole sweep, cutting the SDPA to 32 cores makes it 1.36x "
                       "SLOWER while the product gets 2.10x faster. The lever is class-specific "
                       "even within one architecture, so it is not 'use fewer cores'.",
        },
        "what_blackhole_actually_has": {
            "arch": tm["meta"]["arch"], "grid": tm["meta"]["grid"], "card": tm["meta"]["card"],
            "host": tm["meta"]["host"], "loadavg": tm["meta"]["loadavg"], "recorded_clock": None,
            "core_count_sweep_exists": False,
            "every_arm_ran_the_full_grid": True,
            "trimul": bh(tm, "trimul"),
            "triatt": bh(ta, "triatt"),
            "dense_cube_same_session": cube,
            "trimul_vs_cube_rate_ratio": cube["TFLOPs_at_min"] / bh(tm, "trimul")["TFLOPs_at_min"],
            "triatt_vs_cube_rate_ratio": cube["TFLOPs_at_min"] / bh(ta, "triatt")["TFLOPs_at_min"],
        },
        "conclusion": [
            "The ranked claim is a Wormhole B0 measurement on an 8x9 grid of 72 cores. Blackhole "
            "runs 11x10 = 110 cores here and 13x10 elsewhere, with different DRAM behaviour.",
            "No Blackhole core-count sweep exists for either class: every Blackhole arm ran the "
            "full grid. The 810 Mcycle estimate is therefore a cross-architecture transfer.",
            "This project has been burned by exactly that before: a Blackhole DRAM roof inverted "
            "an L1-residency lever's sign, and a Blackhole-fitted envelope capped Wormhole Galaxy.",
            "Blackhole does show a large same-session efficiency gap at 512 aa: trimul reaches "
            "%.2f TFLOP/s and triatt %.2f against a dense cube's %.2f on the same card in the "
            "same session. That gap is measured on the right architecture and is the honest "
            "motivation, though a trimul is not a dense cube and not all of it is recoverable."
            % (bh(tm, "trimul")["TFLOPs_at_min"], bh(ta, "triatt")["TFLOPs_at_min"],
               cube["TFLOPs_at_min"]),
            "Both Blackhole sessions ran at loadavg 4.6 to 6.2 with no recorded clock, like every "
            "other measurement in this corpus.",
        ],
    }
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
