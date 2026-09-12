#!/usr/bin/env python3
"""Run the identical tile census against the WORMHOLE capture, as a cross-architecture control.

`census_tiles.py` answers the question on Blackhole. This runs the same code, unchanged, on
`b2z2-whglx-profiler-build`'s whglx capture of the same 512 aa PairformerLayer: same model, same
272 programs, same shapes, different silicon (WH, 8x9 = 72 cores, ~1 GHz, 239.8 GB/s measured
DRAM roof against BH's 11x10 = 110, 1.35 GHz, 444.9 GB/s).

If the input-tile wait were a tile-count term, a tile model would fit on at least one of them.
It fits on neither, with almost the same negative R2, and a byte model outranks it by almost the
same margin on both. That is the point of the control.

Fetch the input with:

    git show origin/wk/b2z2-whglx-profiler-build:perf/b2z2_profiler/block_prof/reports/\\
      2026_09_12_15_16_25/ops_perf_results_2026_09_12_15_16_25.csv.gz > <dir>/wh.csv.gz
    git show origin/wk/b2z2-whglx-profiler-build:perf/b2z2_profiler/block_census_whglx_c1.json \\
      > <dir>/wh_census.json

Usage: wh_crosscheck.py <dir> <outdir>
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import census_tiles as C                                                       # noqa: E402

ROOF_WH_GBS = 239.8      # MEASURED WH DRAM roof, b2z-arch-deficit, whglx card 28
NS_PER_TILE_WH = 71.3    # MEASURED WH tile pass, the constant wave 1 applied to a BH cell


def main():
    src, outdir = Path(sys.argv[1]), Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    cen, rows = C.window(src / "wh.csv.gz", src / "wh_census.json")
    d = C.analyse("PairformerLayer-WH", cen, rows)
    t = d["total"]
    S = [s for s in d["sites"] if s["trisc1_ns"] > 0]
    y = [s["wait_ns"] for s in S]
    L = [f"=== PairformerLayer on WORMHOLE: {d['programs']} programs, span {d['span_ms']:.4f} ms",
         "    whglx card 1, 8x9 = 72 cores, profiler-enabled source build (b2z2-whglx-profiler-build)",
         f"  CB tile arrivals per core                  {t['arrivals_pc']:12,.0f}",
         f"  DRAM read {t['dram_rd']/1e6:9,.1f} MB    L1 read {t['l1_rd']/1e6:9,.1f} MB",
         f"  MEASURED CB wait-front/core {t['wait_ns']/1e6:.4f} ms of TRISC1 {t['trisc1_ns']/1e6:.4f} ms"
         f"  = {100*t['wait_ns']/t['trisc1_ns']:.1f} %",
         f"    (b2z2-whglx-profiler-build publishes 47.4499 ms = 62.0 %; this independent window"
         f" reads {t['wait_ns']/1e6:.4f} = {100*t['wait_ns']/t['trisc1_ns']:.1f} %)",
         f"  arrivals x {NS_PER_TILE_WH} ns/tile (the WH rate) = {t['arrivals_pc']*NS_PER_TILE_WH/1e6:.4f} ms"
         f"  = {100*t['arrivals_pc']*NS_PER_TILE_WH/t['wait_ns']:.1f} % of the WH wait",
         f"  DRAM read at the {ROOF_WH_GBS} GB/s WH roof = {t['dram_rd']/(ROOF_WH_GBS*1e9)*1e3:.4f} ms"
         f"  = {100*t['dram_rd']/(ROOF_WH_GBS*1e9)*1e9/t['wait_ns']:.1f} % of the wait",
         "",
         f"  What predicts the per-site wait? {len(S)} sites that run a compute kernel:"]
    for name, fn in (("tile arrivals only", lambda s: [s["arrivals_pc"]]),
                     ("tile-pair MACs only", lambda s: [s["macs_pc"]]),
                     ("DRAM bytes only", lambda s: [s["dram_rd"]]),
                     ("DRAM + L1 bytes", lambda s: [s["dram_rd"], s["l1_rd"]]),
                     ("DRAM + L1 + per-program", lambda s: [s["dram_rd"], s["l1_rd"], s["n"]])):
        A = [fn(s) for s in S]
        x = C.lstsq(A, y)
        L.append(f"    R2 = {C.r2(A, y, x):7.4f}   {name:26s} " + "  ".join(f"{c:.5g}" for c in x))
    A = [[s["dram_rd"], s["l1_rd"]] for s in S]
    x = C.lstsq(A, y)
    L.append(f"  two-bandwidth fit: DRAM {1/x[0]:,.1f} GB/s ({100/x[0]/1e9/ROOF_WH_GBS*1e9:.0f} %"
             f" of the {ROOF_WH_GBS} roof), L1-interleaved {1/x[1]:,.1f} GB/s")
    L.append("  Spearman against the measured per-site wait:"
             f"  bytes {C.spearman(y, [s['dram_rd']+s['l1_rd'] for s in S]):+.3f}"
             f"   tile arrivals {C.spearman(y, [s['arrivals_pc'] for s in S]):+.3f}"
             f"   programs {C.spearman(y, [s['n'] for s in S]):+.3f}")
    txt = "\n".join(L)
    (outdir / "WH-CROSSCHECK.txt").write_text(txt + "\n")
    json.dump(d, open(outdir / "wh_tile_census.json", "w"), indent=1)
    print(txt)


if __name__ == "__main__":
    main()
