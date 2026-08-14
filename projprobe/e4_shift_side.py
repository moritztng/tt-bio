#!/usr/bin/env python3
"""E4 -- which side carries the translation, and it is not the side the plan assumed.

THE OBSERVATION. estep_e2e.py measured the phase-shift stack at 47.4% of a composed iteration, the
largest single term, and section 10.1 of the deliverable GO'd a fused kernel to take it from 628.9 ms
to a ~180 ms write-bound floor. That is a 3.5x on the term. But the term's SIZE is a design choice,
not a constant, and the plan never examined it.

The cross term is symmetric in which operand carries the phase:

    C[p,t,o] = Re sum_j  X_p(j) e^{-i phi_t(j)} conj(A_o(j))
             = Re sum_j  X_p(j) conj( A_o(j) e^{+i phi_t(j)} )

Shifting the PARTICLE builds a stack of B*N_t rows once per block, so per iteration it builds
N_p * N_t * N_pix elements. Shifting the REFERENCE builds N_o * N_t * N_pix elements ONCE per
iteration, because the reference stack is shared by every particle in every block. The ratio is

    N_o / N_p  =  1152 / 4452  =  0.259

so the reference side is 3.86x less traffic at the tutorial's own parameters. This is exact, not an
approximation: |e^{-i phi}| = 1 so the |X|^2 self term is translation-invariant either way, |A_o|^2 is
untouched, and the CTF and the per-shell 1/(2 sigma^2) weights fold into either operand identically.

WHAT COULD MAKE IT NOT PAY, and this is what the screen is for:
  1. The GEMM shape flips. Shifting X gives m = B*N_t, n = N_o. Shifting A gives m = B,
     n = N_o*N_t -- a much narrower m and a much wider n. E0 measured narrow-m arms losing up to 3x.
  2. The reference stack has to be built in the TRANSPOSED layout the GEMM wants, [N_pix, N_o*N_t],
     because a transpose of a multi-GB tensor would eat the entire saving.
  3. It is 3.86x less traffic but the tensor is bigger at any instant: N_o*N_t*N_pix at once rather
     than B*N_t*N_pix. That is a residency question, not a traffic question.

This screen measures 1 and 3 and leaves 2 as the build risk it is. It does NOT build the reformulation
-- per the method, screen the actual change and predict the landing before building anything.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent
MM_ROOF = 254.5e12
WRITE_ROOF = 173.5e9
MIX_ROOF = 420.2e9


def cfg():
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                           math_approx_mode=False, fp32_dest_acc_en=True,
                                           packer_l1_acc=True)


def best_of(dev, fn, reps):
    outs = fn()
    ttnn.synchronize_device(dev)
    for o in outs:
        ttnn.deallocate(o)
    walls = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        outs = fn()
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - t0)
        for o in outs:
            ttnn.deallocate(o)
    return min(walls), max(walls)


def stack6(ar, ai, cr, ci):
    """The six eltwise ops that build a phase-shifted stack, either side."""
    p1, p2 = ttnn.multiply(ar, cr), ttnn.multiply(ai, ci)
    sr = ttnn.subtract(p1, p2)
    ttnn.deallocate(p1); ttnn.deallocate(p2)
    p3, p4 = ttnn.multiply(ar, ci), ttnn.multiply(ai, cr)
    si = ttnn.add(p3, p4)
    ttnn.deallocate(p3); ttnn.deallocate(p4)
    return [sr, si]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pix", type=int, default=13312)
    ap.add_argument("--n-trans", type=int, default=36)
    ap.add_argument("--n-orient", type=int, default=1152)
    ap.add_argument("--n-part", type=int, default=4452)
    ap.add_argument("-B", "--block", type=int, default=64)
    ap.add_argument("--bigb", type=int, default=1024, help="the particle block the A-side GEMM wants")
    ap.add_argument("--no-chunk", type=int, default=18432,
                    help="orientation*translation columns per A-side GEMM, so operands fit DRAM")
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()

    npix, nt, no, npart, B = a.n_pix, a.n_trans, a.n_orient, a.n_part, a.block
    ntp = 32 * ((nt + 31) // 32)          # tile-align the translation axis: worth 2.44x, section 4.1
    dev = ttnn.open_device(device_id=0)
    res = dict(vars(a)); res["nt_padded"] = ntp
    try:
        mk = lambda *s: ttnn.from_torch(torch.randn(*s, dtype=torch.bfloat16),
                                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        c = cfg()
        print(f"N_pix={npix} N_t={nt}->{ntp} N_o={no} N_p={npart}", flush=True)

        # ---- 1. the eltwise rate, at both stacks' sizes -----------------------------------------
        # The saving is an ELEMENT COUNT, so the only thing to measure is whether the per-element
        # rate holds at the bigger tensor. If it does, the 3.86x is arithmetic.
        print("\n1. the six-op stack build, per-element rate at both sizes", flush=True)
        el_rows = []
        for tag, nrow in (("X-side, one block", B), ("A-side, an orientation chunk", 256)):
            xr, xi = mk(1, nrow, ntp, npix), mk(1, nrow, ntp, npix)
            cr, ci = mk(1, 1, ntp, npix), mk(1, 1, ntp, npix)
            lo, hi = best_of(dev, lambda: stack6(xr, xi, cr, ci), a.reps)
            for t in (xr, xi, cr, ci):
                ttnn.deallocate(t)
            nel = nrow * ntp * npix
            by = 6 * 3 * nel * 2
            rec = dict(tag=tag, nrow=nrow, n_elem=nel, wall_s=lo, spread=(hi - lo) / lo,
                       gbs=by / lo / 1e9, ns_per_elem=lo * 1e9 / nel)
            el_rows.append(rec)
            print("   %-30s rows=%-5d %.3e elem  %8.3f ms  %6.1f GB/s  %.4f ns/elem  spread %.1f%%"
                  % (tag, nrow, nel, lo * 1e3, rec["gbs"], rec["ns_per_elem"],
                     100 * rec["spread"]), flush=True)
        res["eltwise"] = el_rows
        ns_el = el_rows[1]["ns_per_elem"]

        # ---- 2. the two GEMM shapes -------------------------------------------------------------
        print("\n2. the compare GEMM, both formulations. Same FLOP, different shape.", flush=True)
        gemms = []
        for tag, mm, kk, nn in (("X-side: m=B*N_t, n=N_o", B * ntp, npix, no),
                                ("X-side, B=1024", a.bigb * ntp // 16, npix, no),
                                ("A-side: m=B, n=N_o*N_t", a.bigb, npix, a.no_chunk)):
            try:
                x = mk(1, 1, mm, kk)
                y = mk(1, 1, kk, nn)
                lo, hi = best_of(dev, lambda: [ttnn.matmul(x, y, compute_kernel_config=c)], a.reps)
                ttnn.deallocate(x); ttnn.deallocate(y)
                fl = 2.0 * mm * kk * nn
                rec = dict(tag=tag, m=mm, k=kk, n=nn, wall_s=lo, spread=(hi - lo) / lo,
                           tflops=fl / lo / 1e12, pct_roof=100 * fl / lo / MM_ROOF)
                print("   %-26s m=%-6d k=%-6d n=%-6d %8.3f ms  %7.1f TFLOP/s  %3.0f%% roof"
                      "  spread %.1f%%"
                      % (tag, mm, kk, nn, lo * 1e3, rec["tflops"], rec["pct_roof"],
                         100 * rec["spread"]), flush=True)
            except Exception as e:
                rec = dict(tag=tag, m=mm, k=kk, n=nn, error=str(e)[:200])
                print("   %-26s m=%-6d k=%-6d n=%-6d FAILED: %s"
                      % (tag, mm, kk, nn, str(e)[:120]), flush=True)
            gemms.append(rec)
        res["gemms"] = gemms

        # ---- 3. the per-iteration comparison ----------------------------------------------------
        print("\n3. per iteration, the two formulations", flush=True)
        el_x = npart * ntp * npix          # stack elements built per iteration, X side
        el_a = no * ntp * npix             # ... A side, built ONCE
        s_x, s_a = el_x * ns_el / 1e9, el_a * ns_el / 1e9
        # the fused floor for each: read the operand and the ramps once, write the stack once
        fl_x = npart / B * (2 * B * npix * 2 + 2 * ntp * npix * 2 + 2 * B * ntp * npix * 2) / WRITE_ROOF
        fl_a = (2 * no * npix * 2 + 2 * ntp * npix * 2 + 2 * no * ntp * npix * 2) / WRITE_ROOF
        ok = [g for g in gemms if "tflops" in g]
        gx = next((g for g in ok if g["tag"].startswith("X-side: ")), None)
        ga = next((g for g in ok if g["tag"].startswith("A-side")), None)
        c_flop = 2 * 2 * npart * ntp * npix * no
        cx = c_flop / (gx["tflops"] * 1e12) if gx else float("nan")
        ca = c_flop / (ga["tflops"] * 1e12) if ga else float("nan")
        res["iteration"] = dict(el_x=el_x, el_a=el_a, elem_ratio=el_x / el_a,
                                S_x_s=s_x, S_a_s=s_a, S_x_fused_floor_s=fl_x,
                                S_a_fused_floor_s=fl_a, C_x_s=cx, C_a_s=ca,
                                stack_resident_x_mb=2 * B * ntp * npix * 2 / 2 ** 20,
                                stack_resident_a_mb=2 * no * ntp * npix * 2 / 2 ** 20)
        print("   stack elements per iteration:  X side %.3e   A side %.3e   ratio %.2fx"
              % (el_x, el_a, el_x / el_a), flush=True)
        print("   term S, composite:             X %8.1f ms   A %8.1f ms" % (s_x * 1e3, s_a * 1e3),
              flush=True)
        print("   term S, fused floor:           X %8.1f ms   A %8.1f ms" % (fl_x * 1e3, fl_a * 1e3),
              flush=True)
        print("   term C, at each shape's rate:  X %8.1f ms   A %8.1f ms" % (cx * 1e3, ca * 1e3),
              flush=True)
        print("   S+C composite:                 X %8.1f ms   A %8.1f ms" %
              ((s_x + cx) * 1e3, (s_a + ca) * 1e3), flush=True)
        print("   stack resident at once:        X %8.0f MB  A %8.0f MB"
              % (res["iteration"]["stack_resident_x_mb"],
                 res["iteration"]["stack_resident_a_mb"]), flush=True)

        out = HERE / f"e4_shift_side_npix{npix}.json"
        out.write_text(json.dumps(res, indent=1))
        print("\nwrote", out, flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
