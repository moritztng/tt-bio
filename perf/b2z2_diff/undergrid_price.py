#!/usr/bin/env python3
"""What the diffusion step's two under-filled sites cost, against both Blackhole roofs.

75.74 % of the grid is busy during a diffusion step and 99 % of the shortfall is two sites, both
pinned to 16 of 110 cores because 512 tokens is 16 tile rows and both programs are row-parallel.
This prices them against the two floors the campaign has measured on Blackhole, so the lever has a
mechanism rather than a hope:

  datum rate   20.82 ns/tile binary, 30.44 ns/tile unary copy (`b2z2-datum-rate-floor`, measured on
               the cell at 1350 MHz; the retired 71.3 ns/tile was a Wormhole number)
  DRAM         444.9 GB/s measured (`b2z-arch-deficit`)

Inputs are the per-site census from `b2z2-tile-shape-and-format` (`perf/b2z2_layout/PER-SITE-TABLE.md`,
DiffusionStep section, current build at N_padded 4480). No device time is spent here.
"""
from __future__ import annotations

import json
from pathlib import Path

TILE_BYTES = 32 * 32 * 2          # bf16
DRAM_ROOF_GBS = 444.9
STEPS_PER_FOLD = 200

# (site, programs/step, kernel ms/step, cores, tiles in, tiles out, datum floor ns/tile)
SITES = [
    ("NlpCreateHeads 1x1x512x3072 -> 1x16x512x64", 24, 1.0215, 16, 16 * 96, 3 * 16 * 16 * 2, 30.44),
    ("LayerNorm 1x1x512x768", 100, 1.5942, 16, 16 * 24, 16 * 24, 20.82),
]


def main():
    out = {"dram_roof_gbs": DRAM_ROOF_GBS, "steps_per_fold": STEPS_PER_FOLD, "sites": []}
    total_at_roof = 0.0
    for name, n, ms, cores, t_in, t_out, floor_ns in SITES:
        us = ms / n * 1e3
        floor_us = t_in / cores * floor_ns / 1e3
        byt = (t_in + t_out) * TILE_BYTES
        gbs = byt / (us * 1e-6) / 1e9
        at_roof_us = byt / (DRAM_ROOF_GBS * 1e9) * 1e6
        saved_s = (us - at_roof_us) * n * STEPS_PER_FOLD / 1e6
        total_at_roof += saved_s
        row = {"site": name, "programs_per_step": n, "us_per_program": round(us, 2),
               "cores": cores, "tiles_in": t_in, "tiles_out": t_out,
               "datum_floor_us": round(floor_us, 2), "x_over_datum_floor": round(us / floor_us, 1),
               "MB": round(byt / 1e6, 2), "achieved_gbs": round(gbs, 1),
               "pct_of_dram_roof": round(gbs / DRAM_ROOF_GBS * 100, 1),
               "us_at_roof": round(at_roof_us, 2), "s_per_fold_if_at_roof": round(saved_s, 3)}
        out["sites"].append(row)
        print(json.dumps(row), flush=True)
    out["total_s_per_fold_if_at_roof"] = round(total_at_roof, 3)
    print("total if both reached the DRAM roof: %.3f s/fold" % total_at_roof)
    p = Path(__file__).with_name("undergrid_price.json")
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)


if __name__ == "__main__":
    main()
