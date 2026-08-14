#!/usr/bin/env python3
"""E1 -- the two screens the composed floor cannot be written without.

E0 (projprobe/e0_gemm_roof.py) measured the bf16 matmul roof at 249.1 TFLOP/s HiFi2 and showed the
compare GEMM reaching it at m=6400,k=8192,n=4096. Two things it left open, both pre-registered as
gates in the plan:

  E1.2  fp32_dest_acc_en.  E0 ran with fp32_dest_acc_en=False. The compare accumulates over k up to
        13,312 terms and relion-backprojection.md section 14 measured a bf16 DESTINATION accumulator
        producing a 1.0% SYSTEMATIC SCALE BIAS at depth 48 -- bias, not noise. So the accumulator is
        not a free choice here either. This arm reports rate AND relative L2 against an fp64 torch
        reference for both settings, at the deepest k the crop reaches.

  E1.3  the phase-shift stack.  Section 2.3 of the plan writes the floor as five terms and omits the
        construction of Xs : [B*N_t, N_pix] from X : [B, N_pix] by complex multiply against N_t
        precomputed phase ramps. It is eltwise, it is the compare's own input, and nothing has priced
        it. If it is above 10% of the compare the floor is amended.

        The term matters at a specific operating point and not others, which is the point of measuring
        it rather than asserting it: the stack is built ONCE per particle block and reused across ALL
        N_o orientations, so its weight relative to the compare falls as 1/N_o. It is largest at a
        WIDE resolution crop with a NARROW orientation block.

Both arms: warm launch, best of N inside one process, device synchronised at both ends. Never an
isolated per-op timing (memory: tt-bio-isolated-op-timing-oversync-inflates-cost).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent

# Measured on THIS card by projprobe/e0_gemm_roof.py, HiFi2, best of 5 under benchlock.
MM_ROOF = 249.1e12
# Measured on THIS card by projprobe/b0_roofs.py.
READ_ROOF = 404.5e9
WRITE_ROOF = 173.5e9
MIX_ROOF = 420.2e9


def cfg(fid=ttnn.MathFidelity.HiFi2, fp32acc=False):
    return ttnn.WormholeComputeKernelConfig(math_fidelity=fid, math_approx_mode=False,
                                           fp32_dest_acc_en=fp32acc, packer_l1_acc=True)


def best_of(dev, fn, reps):
    """Warm launch, then best-of-reps around one synchronise at each end."""
    out = fn()
    ttnn.synchronize_device(dev)
    if out is not None:
        ttnn.deallocate(out)
    walls = []
    for _ in range(reps):
        ttnn.synchronize_device(dev)
        t0 = time.perf_counter()
        out = fn()
        ttnn.synchronize_device(dev)
        walls.append(time.perf_counter() - t0)
        if out is not None:
            ttnn.deallocate(out)
    return min(walls), max(walls)


# ---------------------------------------------------------------------------------------------
# E1.2  the destination accumulator, rate and accuracy
# ---------------------------------------------------------------------------------------------

def e12(dev, res, reps):
    print("E1.2  the compare GEMM's destination accumulator: rate, and the accuracy at real depth",
          flush=True)
    print("      (bproj section 14 found a bf16 DST accumulator biasing the SCALE by 1.0% at depth 48;"
          "\n       the compare's depth is k, up to 13,312, so this is the same question one axis over)",
          flush=True)
    rows = []
    for (m, k, n) in ((6400, 8192, 4096), (1600, 8192, 18432), (7744, 13312, 2304)):
        # One pair of operands, both accumulator settings, so the comparison is A/A on the data too.
        torch.manual_seed(11)
        ah = torch.randn(1, 1, m, k, dtype=torch.bfloat16)
        bh = torch.randn(1, 1, k, n, dtype=torch.bfloat16)
        a = ttnn.from_torch(ah, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        b = ttnn.from_torch(bh, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        # fp64 reference on a SLICE of the output: the full [m,n] fp64 matmul is too big to hold,
        # and a column slice is the same accumulation depth, which is the axis under test.
        ncol = 128
        ref = (ah[0, 0].to(torch.float64) @ bh[0, 0, :, :ncol].to(torch.float64))
        rn = float(torch.linalg.norm(ref))
        for fp32acc in (False, True):
            c = cfg(fp32acc=fp32acc)
            lo, hi = best_of(dev, lambda: ttnn.matmul(a, b, compute_kernel_config=c), reps)
            out = ttnn.matmul(a, b, compute_kernel_config=c)
            ttnn.synchronize_device(dev)
            got = ttnn.to_torch(out)[0, 0, :, :ncol].to(torch.float64)
            ttnn.deallocate(out)
            rel = float(torch.linalg.norm(got - ref)) / rn
            # A systematic scale bias shows up as a mean ratio away from 1, which a relative L2
            # cannot distinguish from noise. Report it separately -- that is how bproj found its.
            mask = ref.abs() > 1e-9
            scale = float((got[mask] / ref[mask]).mean())
            tf = 2.0 * m * k * n / lo / 1e12
            rec = dict(m=m, k=k, n=n, fp32_dest_acc=fp32acc, wall_s=lo,
                       spread=(hi - lo) / lo, tflops=tf, pct_of_roof=100 * tf * 1e12 / MM_ROOF,
                       rel_l2=rel, mean_scale=scale)
            rows.append(rec)
            print("      m=%-5d k=%-6d n=%-6d fp32dst=%-5s %8.3f ms %7.1f TFLOP/s (%.0f%% roof) "
                  "rel L2 %.2e  mean scale %.6f  spread %.1f%%"
                  % (m, k, n, fp32acc, lo * 1e3, tf, rec["pct_of_roof"], rel, scale,
                     100 * rec["spread"]), flush=True)
        ttnn.deallocate(a)
        ttnn.deallocate(b)
    res["e12"] = rows
    # the gate: fp32 dest costing more than 10% re-opens the precision/perf trade
    for i in range(0, len(rows), 2):
        off, on = rows[i], rows[i + 1]
        cost = 100 * (on["wall_s"] - off["wall_s"]) / off["wall_s"]
        print("      -> m=%-5d k=%-6d fp32 dest costs %+.1f%% of the wall;  rel L2 %.2e -> %.2e;"
              "  scale %.6f -> %.6f" % (off["m"], off["k"], cost, off["rel_l2"], on["rel_l2"],
                                        off["mean_scale"], on["mean_scale"]), flush=True)


# ---------------------------------------------------------------------------------------------
# E1.3  the phase-shift stack, the term the floor omits
# ---------------------------------------------------------------------------------------------

def shift_stack(xr, xi, cr, ci, B, nt, npix):
    """Xs = X (x) ramp, broadcast over the particle axis. Six eltwise ops on [B, nt, npix].

    xr/xi are [B, nt, npix] (the particle already broadcast along the translation axis, which is a
    real materialisation and is timed as part of the term). cr/ci are [1, nt, npix] and broadcast.
    """
    ar = ttnn.multiply(xr, cr)
    br = ttnn.multiply(xi, ci)
    sr = ttnn.subtract(ar, br)
    ttnn.deallocate(ar)
    ttnn.deallocate(br)
    ai = ttnn.multiply(xr, ci)
    bi = ttnn.multiply(xi, cr)
    si = ttnn.add(ai, bi)
    ttnn.deallocate(ai)
    ttnn.deallocate(bi)
    ttnn.deallocate(sr)
    return si


def e13(dev, res, reps):
    print("\nE1.3  the phase-shift stack: [B, N_pix] -> [B*N_t, N_pix], the term section 2.3 omits",
          flush=True)
    rows = []
    B = 64
    for (nt, npix, no) in ((121, 13312, 2304), (121, 1024, 2304), (121, 13312, 73728)):
        torch.manual_seed(23)
        # The particle broadcast along the translation axis IS part of the term's cost, so it is
        # materialised here rather than being smuggled in as a free reshape.
        xr = ttnn.from_torch(torch.randn(1, B, nt, npix, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        xi = ttnn.from_torch(torch.randn(1, B, nt, npix, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        cr = ttnn.from_torch(torch.randn(1, 1, nt, npix, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        ci = ttnn.from_torch(torch.randn(1, 1, nt, npix, dtype=torch.bfloat16),
                             layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        lo, hi = best_of(dev, lambda: shift_stack(xr, xi, cr, ci, B, nt, npix), reps)
        for t in (xr, xi, cr, ci):
            ttnn.deallocate(t)
        nel = B * nt * npix
        # 6 eltwise ops; the two multiplies against the broadcast ramp read it once per (b) slab.
        # Bytes actually moved: each op reads 2 operands and writes 1, at 2 B/element, except the
        # ramp reads which are 1/B of a slab.
        composite_bytes = 6 * 3 * nel * 2
        # The fused form's floor: read X once, read the ramps once, write Xs. Nothing else.
        fused_bytes = 2 * B * npix * 2 + 2 * nt * npix * 2 + 2 * nel * 2
        # what the compare costs for the same block, at the roof
        cmp_flop = 4 * 2 * (B * nt) * npix * no
        cmp_s = cmp_flop / MM_ROOF
        rec = dict(B=B, n_trans=nt, n_pix=npix, n_orient=no, wall_s=lo, spread=(hi - lo) / lo,
                   n_elem=nel, composite_bytes=composite_bytes, fused_bytes=fused_bytes,
                   implied_gbs=composite_bytes / lo / 1e9,
                   fused_floor_s=fused_bytes / WRITE_ROOF,
                   compare_s_at_roof=cmp_s, pct_of_compare=100 * lo / cmp_s,
                   pct_of_compare_if_fused=100 * (fused_bytes / WRITE_ROOF) / cmp_s)
        rows.append(rec)
        print("      B=%d N_t=%d N_pix=%-6d N_o=%-6d  %8.3f ms  %6.1f GB/s implied  spread %.1f%%"
              % (B, nt, npix, no, lo * 1e3, rec["implied_gbs"], 100 * rec["spread"]), flush=True)
        print("        compare for the same block at the roof: %7.3f ms  -> stack is %5.1f%% of it"
              "   (fused floor %.3f ms = %.1f%%)"
              % (cmp_s * 1e3, rec["pct_of_compare"], rec["fused_floor_s"] * 1e3,
                 rec["pct_of_compare_if_fused"]), flush=True)
    res["e13"] = rows


# ---------------------------------------------------------------------------------------------
# E1.5  term D, the weight reduction over the score matrix
# ---------------------------------------------------------------------------------------------

def weights(c, B, nt, no):
    """diff2 -> posterior weight, the reduction the plan prices at 3 passes over the score matrix.

    RELION subtracts the per-particle minimum before exponentiating (its `min_diff2`), which is the
    only numerically safe order and is also what makes this three passes rather than one.
    """
    mn = ttnn.min(c, dim=-1, keepdim=True)
    d = ttnn.subtract(c, mn)
    e = ttnn.exp(d)
    s = ttnn.sum(e, dim=-1, keepdim=True)
    w = ttnn.divide(e, s)
    for t in (mn, d, e, s):
        ttnn.deallocate(t)
    return w


def e15(dev, res, reps):
    print("\nE1.5  term D: the score matrix -> posterior weights, 3 passes over [B*N_t, N_o]",
          flush=True)
    rows = []
    B = 64
    for (nt, no) in ((121, 2304), (121, 18432)):
        torch.manual_seed(37)
        c = ttnn.from_torch(torch.randn(1, 1, B * nt, no, dtype=torch.bfloat16),
                            layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        lo, hi = best_of(dev, lambda: weights(c, B, nt, no), reps)
        ttnn.deallocate(c)
        nel = B * nt * no
        by = 3 * 2 * nel * 2      # 3 passes, read+write, bf16
        rec = dict(B=B, n_trans=nt, n_orient=no, wall_s=lo, spread=(hi - lo) / lo, n_elem=nel,
                   bytes=by, implied_gbs=by / lo / 1e9, floor_s=by / MIX_ROOF)
        rows.append(rec)
        print("      B=%d N_t=%d N_o=%-6d  %.3e scores  %8.3f ms  %6.1f GB/s of a %.1f mixed roof"
              "  (floor %.3f ms, %.0f%%)  spread %.1f%%"
              % (B, nt, no, nel, lo * 1e3, rec["implied_gbs"], MIX_ROOF / 1e9,
                 rec["floor_s"] * 1e3, 100 * rec["floor_s"] / lo, 100 * rec["spread"]), flush=True)
    res["e15"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    dev = ttnn.open_device(device_id=0)
    res = {"mm_roof_tflops": MM_ROOF / 1e12, "write_roof_gbs": WRITE_ROOF / 1e9,
           "mix_roof_gbs": MIX_ROOF / 1e9, "reps": a.reps}
    try:
        if not a.only or "12" in a.only:
            e12(dev, res, a.reps)
        if not a.only or "13" in a.only:
            e13(dev, res, a.reps)
        if not a.only or "15" in a.only:
            e15(dev, res, a.reps)
    finally:
        ttnn.close_device(dev)
    out = HERE / "e1_screens.json"
    out.write_text(json.dumps(res, indent=1))
    print("\nwrote", out)


if __name__ == "__main__":
    main()
