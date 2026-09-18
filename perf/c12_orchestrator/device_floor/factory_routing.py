#!/usr/bin/env python3
"""How much of the fold routes to ttnn's 1D matmul factory, and what crossing families is worth.

`create_matmul_program_config` picks between the 1D systolic and 2D factories on an output
height/width ratio of 8 and nothing else (`matmul_program_config.cpp:28`), then sets the K block
inside the chosen family. A sweep confined to the derived family cannot see the other factory, and
every C12 sweep before `c12-kblock-unlock`'s was so confined.

    ratio = Mt_total / Nt,  Mt_total = prod(out dims but the last) / 32,  Nt = N / 32.  >8 -> 1D.

PASS 18 REVISION. This script's first version (pass 16, 20d460a4e) put the untouched 1D prize at
0.9454 s and the campaign called it "the largest sized opportunity left". Both of its anchors came
from `c12-kblock-unlock`'s pre-production figures and BOTH were superseded when that row concluded:

  * THE NUMERATOR. kblock's +0.2052 s was measured at shapes the fold never issues. Re-swept at the
    shapes it does issue, fc3 fell from 1.2776x to 1.0113x (inside its own A/A floor, NOT A RESULT,
    and it had been the single largest line at +0.1238 s) and fc2 from 1.2593x to 1.1236x. Its
    concluded size is ~0.035 s, below its own 0.10 s kill criterion, with the fold A/B at 1.0104x
    inside the session A/A spread. So the realized recovery rate is 1.80 %, not 10.55 %.

  * THE SHAPE BASIS. The census key `1x16x512x*` is a fiction: TWO concluded rows found from
    different instruments that `[1,16,512,128]` ships at NO size. kblock's `_linear_block_cfg`
    witness recorded no call in the fold with `mt_total` 256 (it issues 752 and 672), and
    `c12-unfused-silu-bh`'s instrumented fold counted 3,762 calls where the census says 8,960. The
    pair track is row-blocked into ten chunks of 47 rows plus one of 42 at 512 aa, and the chunk
    height is SIZE-DEPENDENT: 64 at 298 aa, 47 at 512, 32 at 768. The census conserves total rows
    while splitting each call into 16-row sub-calls, which inflates the call count ~2.4-2.9x and
    deflates `mt_total` by the same factor -- and `mt_total` is exactly what the ratio and
    `per_core_M` derive from.

So the ranked list below is printed at BOTH shapes. The family split is unchanged by the
correction (every affected key is 1D at either shape, so nothing flips), but the REGIME labels move
a long way: the key this script called "ratio 16.00" is really ratio 47, and kblock's 1.2592x
anchor was measured at the 16 that does not exist. Only fc2 and fc3 have ever been tested at their
real shape, and one of the two died there.

The other 29 keys have never been checked for the same defect. `c12-profiled-fold` is the row that
settles it; until then every per-key second here is unverified in the way pass 7 flagged.
"""
import json
import subprocess

CENSUS = "origin/wk/c10-fold-census:perf/c10_fold_census/runs/sweep2/replay.json"
KBLOCK_BASE_S = 1.9449        # census seconds of the four keys kblock priced
KBLOCK_GAIN_PASS16 = 0.2052   # pre-production, at shapes the fold never issues -- SUPERSEDED
KBLOCK_GAIN_FINAL = 0.035     # concluded, production path, real row-block shapes
BEST_RATIO_FAKE = 1.2592      # its best cross-family ratio at the b=16 shape
BEST_RATIO_REAL = 1.1236      # the same key (fc2) re-measured at the real b=47 shape

# The row-blocked pair track, from two concluded witnesses. Census records the chunk height as 16
# rows; the fold issues 47 at 512 aa (ten chunks) plus a 42-row tail (one chunk).
CENSUS_CHUNK_ROWS = 16
REAL_CHUNK_ROWS_512 = 47


def ratio(dims, chunk_rows=None):
    """Mt_total/Nt. chunk_rows rewrites a row-blocked pair key's leading chunk height."""
    dims = list(dims)
    if chunk_rows is not None:
        dims[1] = chunk_rows
    lead = 1
    for x in dims[:-2]:
        lead *= x
    mt, nt = lead * dims[-2] / 32.0, dims[-1] / 32.0
    return (mt, nt, mt / nt) if nt else None


def is_row_blocked(dims):
    """A pair-track key the census recorded at the 16-row chunk height that never ships."""
    return len(dims) == 4 and dims[0] == 1 and dims[1] == CENSUS_CHUNK_ROWS and dims[2] == 512


def main():
    d = json.loads(subprocess.run(["git", "show", CENSUS], capture_output=True,
                                  check=True, text=True).stdout)
    rows = [r for r in d["rows"]
            if r["arm"] in ("linear", "matmul") and not str(r["label"]).endswith("@grid110")]
    tot = {"1D": 0.0, "2D": 0.0}
    flips, keys, fiction_s = 0, [], 0.0
    for r in rows:
        dims = list(r["out"])
        if len(dims) < 2:
            continue
        got = ratio(dims)
        if not got:
            continue
        mt, nt, q = got
        rb = is_row_blocked(dims)
        q_real, mt_real = q, mt
        if rb:
            mt_real, _, q_real = ratio(dims, REAL_CHUNK_ROWS_512)
        s = (r.get("s_per_call_qualified") or 0) * (r["calls"] or 0)
        fac = "1D" if q > 8 else "2D"
        if ("1D" if q_real > 8 else "2D") != fac:
            flips += 1
        if rb:
            fiction_s += s
        tot[fac] += s
        keys.append((s, r["arm"], "x".join(map(str, dims)), r["K"], mt, nt, q, mt_real, q_real,
                     fac, rb))

    print(f"{'key':44}{'Mt_cen':>8}{'Nt':>6}{'ratio':>9}"
          f"{'Mt_real':>9}{'ratio_real':>11}{'fac':>5}{'census s':>10}")
    for s, a, o, K, mt, nt, q, mtr, qr, fac, rb in sorted(keys, key=lambda x: -x[0])[:14]:
        mark = "  <- b=16 FICTION" if rb else ""
        print(f"{a+'|'+o+'|K='+str(K):44}{mt:8.0f}{nt:6.1f}{q:9.2f}"
              f"{mtr:9.0f}{qr:11.2f}{fac:>5}{s:10.4f}{mark}")

    one, two = tot["1D"], tot["2D"]
    untouched = one - KBLOCK_BASE_S
    r16, rfin = KBLOCK_GAIN_PASS16 / KBLOCK_BASE_S, KBLOCK_GAIN_FINAL / KBLOCK_BASE_S
    print(f"\nrouted 1D (ratio > 8), 2D untested : {one:8.4f} s of census seconds")
    print(f"routed 2D (ratio <= 8)             : {two:8.4f} s")
    print(f"keys on the b=16 fiction shape     : {fiction_s:8.4f} s "
          f"({100*fiction_s/(one+two):.1f} % of the class), family flips under correction: {flips}")
    print(f"1D seconds kblock did NOT touch    : {untouched:8.4f} s")
    print(f"\n{'':4}{'anchor':38}{'rate':>8}{'prize':>10}")
    print(f"{'':4}{'pass 16, pre-production kblock':38}{100*r16:7.2f}%{untouched*r16:10.4f} s"
          f"   <- REFUTED, both anchors superseded")
    print(f"{'':4}{'concluded kblock, real shapes':38}{100*rfin:7.2f}%{untouched*rfin:10.4f} s"
          f"   <- the defensible one")
    print(f"{'':4}{'uniform at 1.2592x (b=16 shape)':38}{'':>8}{untouched*(1-1/BEST_RATIO_FAKE):10.4f} s"
          f"   <- measured at a shape that never ships")
    print(f"{'':4}{'uniform at 1.1236x (real shape)':38}{'':>8}{untouched*(1-1/BEST_RATIO_REAL):10.4f} s"
          f"   <- 1 of the 2 keys tested real died")
    print(f"\nHalve any of these again for the ~2x census overprice on L1-resident keys, so the")
    print(f"defensible in-fold prize is about {untouched*rfin/2:.3f} s. Cross-family is NOT the")
    print(f"largest sized opportunity left; on its own anchors it is worth under 0.1 s.")


if __name__ == "__main__":
    main()
