#!/usr/bin/env python3
"""E5 -- the refinement priced over RELION's REAL per-iteration sampling.

Section 6 of the deliverable priced all 15 iterations at N_o = 1152 and was retracted, because the
sampling counts were inferred from RELION's command line instead of read out of its log. RELION prints
NrOrientations and NrTranslations for the coarse and the oversampled pass of every iteration. Those
numbers, read from Refine3D/job019/run.out, are hard-coded below with their source.

WHAT THIS PRICES, and it is not the whole refinement:
  the COARSE pass only -- every orientation x translation evaluated for every particle. That is a
  fully enumerated, well-defined quantity, so it can be priced exactly from measured rates.

WHAT IT DOES NOT PRICE:
  the oversampled (fine) pass, which RELION runs only at the SIGNIFICANT coarse poses. Its sampling
  is 32x the coarse pass but its particle-pose count is N_sig-dependent and N_sig has never been
  measured (deliverable section 9 item 2). Reported as a multiplier, not folded into a total.
  Also not priced: the M-step, I/O, and the inter-card volume reduction.

Every rate below is MEASURED on qb1 card 0 and cited. No rate is a spec sheet and no rate is assumed
constant where it was measured to vary.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ---- MEASURED rates ------------------------------------------------------------------------------
FFT_IMG_S = 616078.0        # ttnn-fft-kernel-spike.md 17.4
PROJ_SLICE_S = 428200.0     # relion-projection-complete.md 3.3
BPROJ_SLICE_S = 413200.0    # relion-backprojection.md 12
NS_PER_STACK_ELEM = 0.0807  # e4_shift_side.py, A-side composite, npix 25088
NS_PER_SCORE = 0.886e6 / 1.784e7   # e1_screens.py term D: 0.886 ms / 1.784e7 scores
WRITE_ROOF = 173.5e9
# A-side compare rates, MEASURED in e4_shift_side.py / e4_transposeb.json. The rate depends on k,
# which is the resolution crop, so it is a function and not a constant -- that is the whole reason
# section 6's single-rate extrapolation was wrong in the first place.
def compare_tflops(k, n, transpose_b):
    """Piecewise-constant on MEASURED anchors, rounded DOWN to the nearest measured shape.

    The compare's rate is a function of BOTH k (the resolution crop) and n (the output width), and
    pricing it with one constant is exactly the mistake that made section 6 wrong. Anchors, all
    measured on qb1 card 0 at HiFi2 with fp32 dest accumulate on:

        k=512    n=1152    30.0 TFLOP/s   e4_shift_side.py, X-side, iteration 1
        k=512    n=18432   54.1           e4_shift_side.py, A-side, iteration 1
        k=25088  n=1160   115.2           e4_shift_side.py, X-side, late iteration
        k=13312  n=2304   175.8           estep_e2e.py, X-side
        k=25088  n=18432  253.4           e4_shift_side.py, A-side, late -- 100% of roof
        the same A-side arm via transpose_b: 169.2, i.e. +48.2%  (e4_transposeb.json)

    Rounding down means every figure below is conservative for the arm being priced.
    """
    if k <= 1024:
        base = 54.1e12 if n >= 4096 else 30.0e12
    elif n < 2048:
        base = 115.2e12
    elif n < 8192:
        base = 175.8e12
    else:
        base = 253.4e12
    return base / 1.482 if transpose_b else base

# ---- RELION's own per-iteration sampling, from Refine3D/job019/run.out --------------------------
# (iteration, CurrentImageSize, coarse N_o, coarse N_t, fine N_o, fine N_t)
TRAJ = [
    (1,  32,  1152, 21,  9216,  84),
    (2,  76,  1152, 21,  9216,  84),
    (3,  80,  9216, 21, 73728,  84),
    (4, 128,  9216, 21, 73728,  84),
    (5, 144,  9216, 21, 73728,  84),
    (6, 146,   145, 29,  1160, 116),
    (7, 146,   145, 29,  1160, 116),
    (8, 150,   145, 29,  1160, 116),
    (9, 154,   145,  9,  1160,  36),
    (10, 154,  145,  9,  1160,  36),
    (11, 160,  145,  9,  1160,  36),
    (12, 160,  145,  9,  1160,  36),
    (13, 196,  145,  9,  1160,  36),
    (14, 196,  145,  9,  1160,  36),
    (15, 196,  145,  9,  1160,  36),
]
NPART = 4452

# MEASURED, per iteration, from RELION's OWN _rlnNrOfSignificantSamples in run_it0NN_data.star
# (projprobe/e6_nsig.json). This closes the deliverable's section 9 item 2, which was the
# highest-value open number in the whole task: term E scales linearly in it and it was a guess of 20.
#
# CAVEAT that makes this conservative: RELION counts significant poses in the FULL hidden-variable
# space, orientation x translation. Term E backprojects once per significant ORIENTATION, because
# translations are applied by phase-shifting the image rather than by a separate slice. So the true
# orientation count is <= this, and term E below is an UPPER bound.
NSIG = {1: 1.0, 2: 192.3, 3: 9.2, 4: 7.0, 5: 2.1, 6: 2.1, 7: 5.2, 8: 2.2,
        9: 2.1, 10: 3.6, 11: 1.9, 12: 3.5, 13: 3.1, 14: 8.0, 15: 8.5}


def npix_of(size):
    """Tile-aligned centred crop, half-space complex pixels."""
    sc = 32 * ((size + 31) // 32)
    return sc * sc // 2


def price(route, shift):
    """shift: 'particle' | 'reference' | 'min' -- 'min' picks the cheaper side PER ITERATION."""
    rows, tot = [], {}
    tb = route == "transpose_b"
    for (it, size, no_c, nt_c, no_f, nt_f) in TRAJ:
        npix = npix_of(size)
        ntp = 32 * ((nt_c + 31) // 32)
        # WHICH SIDE CARRIES THE PHASE is a per-iteration choice, because the stack's row count is
        # N_o on the reference side and N_p on the particle side, and RELION's N_o crosses N_p twice
        # over this trajectory: 1152 -> 9216 -> 145 against a fixed N_p of 4452.
        side = shift if shift != "min" else ("reference" if no_c < NPART else "particle")
        el = (no_c * ntp * npix) if side == "reference" else (NPART * ntp * npix)
        s = el * NS_PER_STACK_ELEM / 1e9
        # term C: identical FLOP either way, but the GEMM's n is N_o*N_t on the reference side and
        # only N_o on the particle side, and narrow n is where the compare loses the roof.
        gemm_n = no_c * ntp if side == "reference" else no_c
        c = 2 * 2 * NPART * ntp * npix * no_c / compare_tflops(npix, gemm_n,
                                                               tb and side == "reference")
        d = NPART * ntp * no_c * NS_PER_SCORE / 1e9
        a = NPART / FFT_IMG_S
        b = no_c / PROJ_SLICE_S
        e = NPART * NSIG[it] / BPROJ_SLICE_S
        r = dict(it=it, size=size, npix=npix, no=no_c, nt=nt_c, ntp=ntp, side=side, gemm_n=gemm_n,
                 A=a, B=b, S=s, C=c, D=d, E=e, total=a + b + s + c + d + e,
                 nsig=NSIG[it], fine_multiplier=(no_f * nt_f) / (no_c * nt_c))
        rows.append(r)
        for k in ("A", "B", "S", "C", "D", "E", "total"):
            tot[k] = tot.get(k, 0.0) + r[k]
    return rows, tot


def main():
    out = {}
    for shift, route, tag in (("particle", "plain", "shift the PARTICLE always (section 4 form)"),
                              ("reference", "transpose_b", "shift the REFERENCE always, transpose_b"),
                              ("min", "transpose_b", "shift the SMALLER side per iteration, transpose_b (TODAY)"),
                              ("min", "plain", "shift the SMALLER side per iteration, stack built transposed")):
        rows, tot = price(route, shift)
        key = f"{shift}_{route}"
        out[key] = dict(rows=rows, total=tot)
        print(f"\n=== {tag} ===", flush=True)
        print("  it  size  N_pix   N_o    side      N_sig     S ms      C ms      E ms   total ms", flush=True)
        for r in rows:
            print("  %2d  %4d %6d %6d  %-9s %6.1f %8.1f  %8.1f  %8.1f   %8.1f"
                  % (r["it"], r["size"], r["npix"], r["no"], r["side"], r["nsig"],
                     r["S"] * 1e3, r["C"] * 1e3, r["E"] * 1e3, r["total"] * 1e3), flush=True)
        print("  TOTAL, 15 iterations, coarse pass, ONE p150:", flush=True)
        print("    A %6.2f s   B %6.2f s   S %6.2f s   C %6.2f s   D %6.2f s   E %6.2f s"
              % (tot["A"], tot["B"], tot["S"], tot["C"], tot["D"], tot["E"]), flush=True)
        print("    => %.2f s  (S %.0f%%, C %.0f%%, E %.0f%%)"
              % (tot["total"], 100 * tot["S"] / tot["total"], 100 * tot["C"] / tot["total"],
                 100 * tot["E"] / tot["total"]), flush=True)

    base = out["particle_plain"]["total"]["total"]
    for k, tag in (("reference_transpose_b", "always reference, transpose_b"),
                   ("min_transpose_b", "smaller side per iteration, transpose_b, TODAY"),
                   ("min_plain", "smaller side per iteration, stack built transposed")):
        t = out[k]["total"]["total"]
        print("\n  %s: %.2f s vs %.2f s -> %.2fx on the whole coarse pass"
              % (tag, t, base, base / t), flush=True)

    # the fine pass, as a multiplier rather than a fabricated total
    rows = out["min_transpose_b"]["rows"]
    print("\n  fine-pass sampling multiplier per iteration (N_o*N_t fine / coarse):", flush=True)
    print("   ", "  ".join("it%d=%.0fx" % (r["it"], r["fine_multiplier"]) for r in rows[:6]),
          "...", flush=True)
    print("    RELION evaluates it only at the significant coarse poses, so the wall-clock",
          flush=True)
    print("    multiplier is N_sig/(N_o*N_t) times that -- and N_sig is UNMEASURED.", flush=True)
    p = HERE / "e5_trajectory.json"
    p.write_text(json.dumps(out, indent=1))
    print("\nwrote", p, flush=True)


if __name__ == "__main__":
    main()
