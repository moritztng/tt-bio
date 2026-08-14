#!/usr/bin/env python3
"""E2 -- the E-step inner loop composed on device, and what the composition costs.

THE LOOP'S REAL SHAPE, which is not the shape the brief assumes. Two of the five terms are
PER-ITERATION and three are PER-PARTICLE-BLOCK, and that split is the whole reason the composed floor
is not a sum of primitive floors:

  per iteration, once          A  transform every particle          N_p images
                               B  project every orientation         N_o slices   <- reused by all p
                               E  backproject the significant ones  N_p * N_sig slices
  per particle block, N_p/B x  S  build the phase-shift stack       B*N_t*N_pix elements
                               C  compare, one GEMM                 [B*N_t, N_pix] x [N_pix, N_o]
                               D  scores -> posterior weights       B*N_t*N_o values

A, B and E have measured rates from the three prior passes and they are per-iteration, so they cannot
be sped up or slowed down by the blocking. S, C and D run once per block with the slice store live
across all of them, and NOTHING has measured them chained. This arm chains them and takes the
difference against E1's isolated measurements of the same three terms. That difference IS the
composition cost.

TWO CORRECTIONS TO THE PLAN'S FLOOR, both from arithmetic rather than measurement:

1. The compare is TWO real matmuls, not four. Re[Xs conj(A)] = Xr.Ar + Xi.Ai. The imaginary part of
   the cross term is never used, because diff2 is real. Plan section 2.3 prices four and section 6.2
   says not to shave it yet; the shave is free and it is exact, so it is taken here and the plan's
   term C is halved.

2. Term S exists. Plan section 2.3 omits it; projprobe/e1_screens.py measured it at 116.9% of the
   compare at N_o = 2304. The floor below has six terms.

DISCIPLINE. One timed region per arm, one synchronise at each end, best of --reps inside one process,
sha256 of the weight matrix on every rep. Per-term cost by ABLATION -- arms that differ by one term,
differenced whole-arm-wall to whole-arm-wall -- never an isolated per-op timing, which over-syncs and
inflates cost about 2x on this fleet (memory: tt-bio-isolated-op-timing-oversync-inflates-cost).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
import ttnn

HERE = Path(__file__).resolve().parent

# Roofs, every one measured on THIS card. e0_gemm_roof.py / e1_screens.py / b0_roofs.py.
# 249.1 is what e0 measured on a square arm; e1 then reached 254.5 on an E-step shape, so the
# matmul roof is a LOWER BOUND and the higher of the two is the honest one to divide by.
MM_ROOF = 254.5e12
WRITE_ROOF = 173.5e9
MIX_ROOF = 420.2e9
# Primitive rates, measured by the three prior passes.
FFT_IMG_S = 616078.0          # ttnn-fft-kernel-spike.md section 17.4
PROJ_SLICE_S = 428200.0       # relion-projection-complete.md section 3.3
BPROJ_SLICE_S = 413200.0      # relion-backprojection.md section 12


def cfg(fp32acc=True):
    """fp32_dest_acc_en ON. e1_screens.py measured it FREE at this shape (-0.2%) and measured the
    bf16 destination accumulator biasing the score scale by +4.7% at k = 13,312."""
    return ttnn.WormholeComputeKernelConfig(math_fidelity=ttnn.MathFidelity.HiFi2,
                                           math_approx_mode=False, fp32_dest_acc_en=fp32acc,
                                           packer_l1_acc=True)


# ---------------------------------------------------------------------------------------------
# the three per-block terms
# ---------------------------------------------------------------------------------------------

def term_s(xr, xi, cr, ci):
    """S: Xs = X (x) ramp. Six eltwise ops, the ramp broadcast over the particle axis."""
    ar, br = ttnn.multiply(xr, cr), ttnn.multiply(xi, ci)
    sr = ttnn.subtract(ar, br)
    ttnn.deallocate(ar); ttnn.deallocate(br)
    ai, bi = ttnn.multiply(xr, ci), ttnn.multiply(xi, cr)
    si = ttnn.add(ai, bi)
    ttnn.deallocate(ai); ttnn.deallocate(bi)
    return sr, si


def term_c(sr, si, ar_t, ai_t, c, m, npix):
    """C: the compare. TWO real matmuls -- diff2 is real, so the cross term's imaginary part is dead.

    The score is the NEGATIVE cross term plus the two self terms; the self terms are O(N_pix) per
    (p,t) and per o rather than O(N_pix N_o), so they are not in this GEMM and not in the floor.

    THE RESHAPE IS THE WHOLE BALLGAME AND IT IS FREE ONLY WHEN N_t IS TILE-ALIGNED. The stack is
    built as [1, B, N_t, N_pix] because the ramp broadcasts over the particle axis, and the GEMM
    wants [1, 1, B*N_t, N_pix]. When N_t = 121 the tiled layout pads each particle's slab to 128
    rows, so collapsing the two leading axes has to physically strip 7 rows of pad per particle --
    a full relayout of a 393 MB tensor, twice. Measured: term C at 68.0 TFLOP/s, 27% of roof,
    against the 221.3 TFLOP/s e1_screens.py measured on the identical matmul shape. Padding N_t to
    128 makes the same reshape a no-op on the tile grid and hands the GEMM its measured rate. The
    5.8% of the compare spent on 7 dead translations buys back a 3.1x.
    """
    sr2 = ttnn.reshape(sr, (1, 1, m, npix))
    si2 = ttnn.reshape(si, (1, 1, m, npix))
    p1 = ttnn.matmul(sr2, ar_t, compute_kernel_config=c)
    p2 = ttnn.matmul(si2, ai_t, compute_kernel_config=c)
    out = ttnn.add(p1, p2)
    ttnn.deallocate(p1); ttnn.deallocate(p2)
    return out


def term_d(score):
    """D: diff2 -> posterior weight. RELION's order: subtract the per-particle minimum, then exp."""
    mn = ttnn.min(score, dim=-1, keepdim=True)
    d = ttnn.subtract(score, mn)
    e = ttnn.exp(d)
    s = ttnn.sum(e, dim=-1, keepdim=True)
    w = ttnn.divide(e, s)
    for t in (mn, d, e, s):
        ttnn.deallocate(t)
    return w


# ---------------------------------------------------------------------------------------------

def floor_terms(np_, no, nt, npix, nsig, b):
    """The six-term composed floor, every roof named and measured. Returns seconds per term."""
    nblock = np_ / b
    # S: the fused form's write-bound floor. Reads X and the ramps once, writes Xs.
    s_bytes = nblock * (2 * b * npix * 2 + 2 * nt * npix * 2 + 2 * b * nt * npix * 2)
    # C: two real matmuls over the whole block set.
    c_flop = 2 * 2 * np_ * nt * npix * no
    # D: three passes over the score matrix.
    d_bytes = 3 * 2 * np_ * nt * no * 2
    return {
        "A_fft_s": np_ / FFT_IMG_S,
        "B_proj_s": no / PROJ_SLICE_S,
        "S_shift_s": s_bytes / WRITE_ROOF,
        "C_compare_s": c_flop / MM_ROOF,
        "D_weights_s": d_bytes / MIX_ROOF,
        "E_bproj_s": np_ * nsig / BPROJ_SLICE_S,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--box", type=int, default=256)
    ap.add_argument("-B", "--block", type=int, default=64)
    ap.add_argument("--n-trans", type=int, default=121)
    ap.add_argument("--nt-pad", type=int, default=0,
                    help="rows the translation axis is padded to; 0 = round N_t up to 32. "
                         "Pass --nt-pad 121 to reproduce the unaligned arm.")
    ap.add_argument("--n-pix", type=int, default=13312, help="tile-aligned resolution crop")
    ap.add_argument("--n-orient", type=int, default=2304)
    ap.add_argument("--n-part", type=int, default=4500)
    ap.add_argument("--n-sig", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    B, nt, npix, no = a.block, a.n_trans, a.n_pix, a.n_orient
    ntp = a.nt_pad or 32 * ((nt + 31) // 32)
    m = B * ntp
    dev = ttnn.open_device(device_id=0)
    res = {k: v for k, v in vars(a).items()}
    res.update({"m": m, "nt_padded": ntp, "mm_roof_tflops": MM_ROOF / 1e12,
                "pad_waste_pct": 100 * (ntp - nt) / nt})
    try:
        torch.manual_seed(101)
        mk = lambda *s: ttnn.from_torch(torch.randn(*s, dtype=torch.bfloat16),
                                        layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
        # the particle block, already broadcast along the translation axis
        xr, xi = mk(1, B, ntp, npix), mk(1, B, ntp, npix)
        # the N_t phase ramps, shared by every particle in the block
        cr, ci = mk(1, 1, ntp, npix), mk(1, 1, ntp, npix)
        # THE SLICE STORE, live across every block: [N_pix, N_o] per component, DRAM resident.
        # Term B produced it once per iteration; the composition's job is to not rebuild it.
        ar_t, ai_t = mk(1, 1, npix, no), mk(1, 1, npix, no)
        store_mb = 2 * npix * no * 2 / 2 ** 20
        res["slice_store_mb"] = store_mb
        res["shift_stack_mb"] = 2 * m * npix * 2 / 2 ** 20
        res["score_matrix_mb"] = m * no * 2 / 2 ** 20
        print(f"box {a.box}  B={B} N_t={nt}->{ntp} N_pix={npix} N_o={no}  m={m}"
              f"  (pad waste {res['pad_waste_pct']:.1f}%)", flush=True)
        print(f"  slice store {store_mb:.0f} MB (live across the whole block)   shift stack "
              f"{res['shift_stack_mb']:.0f} MB   score matrix {res['score_matrix_mb']:.0f} MB",
              flush=True)

        c = cfg(True)

        def arm_s():
            sr, si = term_s(xr, xi, cr, ci)
            return [sr, si]

        def arm_sc():
            sr, si = term_s(xr, xi, cr, ci)
            sc = term_c(sr, si, ar_t, ai_t, c, m, npix)
            ttnn.deallocate(sr); ttnn.deallocate(si)
            return [sc]

        def arm_scd():
            sr, si = term_s(xr, xi, cr, ci)
            sc = term_c(sr, si, ar_t, ai_t, c, m, npix)
            ttnn.deallocate(sr); ttnn.deallocate(si)
            w = term_d(sc)
            ttnn.deallocate(sc)
            return [w]

        def arm_c_only():
            """C and D off a PRE-BUILT stack: the compare with the composition removed, so the
            difference against arm_sc is S's cost inside the chain rather than beside it."""
            sc = term_c(xr, xi, ar_t, ai_t, c, m, npix)
            return [sc]

        arms = {"S": arm_s, "S+C": arm_sc, "S+C+D": arm_scd, "C_only": arm_c_only}
        timings, shas = {}, {}
        for name, fn in arms.items():
            outs = fn()
            ttnn.synchronize_device(dev)
            for o in outs:
                ttnn.deallocate(o)
            walls = []
            sh = set()
            for _ in range(a.reps):
                ttnn.synchronize_device(dev)
                t0 = time.perf_counter()
                outs = fn()
                ttnn.synchronize_device(dev)
                walls.append(time.perf_counter() - t0)
                # sha the LAST output: an A/A on wall time alone is not an A/A (proj section 9).
                sh.add(hashlib.sha256(
                    ttnn.to_torch(outs[-1]).to(torch.float32).numpy().tobytes()).hexdigest())
                for o in outs:
                    ttnn.deallocate(o)
            timings[name] = dict(wall_s=min(walls), all_s=walls,
                                 aa_pct=100 * (max(walls) - min(walls)) / min(walls))
            shas[name] = dict(sha256=sorted(sh)[0], stable=len(sh) == 1)
            print("  arm %-7s %8.3f ms   A/A %4.1f%%   sha %s stable=%s"
                  % (name, min(walls) * 1e3, timings[name]["aa_pct"], sorted(sh)[0][:16],
                     shas[name]["stable"]), flush=True)
        res["arms"], res["shas"] = timings, shas

        # --- the ablation decomposition -------------------------------------------------------
        w_s = timings["S"]["wall_s"]
        w_sc = timings["S+C"]["wall_s"]
        w_scd = timings["S+C+D"]["wall_s"]
        w_c = timings["C_only"]["wall_s"]
        dec = {"S_in_chain_s": w_s, "C_in_chain_s": w_sc - w_s, "D_in_chain_s": w_scd - w_sc,
               "C_alone_s": w_c, "block_total_s": w_scd}
        # what the parts cost measured APART, from e1_screens.json if it is there
        e1p = HERE / "e1_screens.json"
        if e1p.exists():
            e1 = json.loads(e1p.read_text())
            iso_s = next((r["wall_s"] for r in e1.get("e13", [])
                          if r["n_pix"] == npix and r["n_trans"] == nt), None)
            iso_d = next((r["wall_s"] for r in e1.get("e15", [])
                          if r["n_orient"] == no and r["n_trans"] == nt), None)
            dec["S_isolated_s"], dec["D_isolated_s"] = iso_s, iso_d
        res["decomposition"] = dec
        print("\nDECOMPOSITION by ablation (whole-arm walls differenced, never per-op):", flush=True)
        print("  S  %8.3f ms" % (dec["S_in_chain_s"] * 1e3)
              + ("   isolated %8.3f ms  -> composition %+.1f%%"
                 % (dec["S_isolated_s"] * 1e3,
                    100 * (dec["S_in_chain_s"] - dec["S_isolated_s"]) / dec["S_isolated_s"])
                 if dec.get("S_isolated_s") else ""), flush=True)
        print("  C  %8.3f ms   alone off a pre-built stack %8.3f ms  -> chained %+.1f%%"
              % (dec["C_in_chain_s"] * 1e3, w_c * 1e3,
                 100 * (dec["C_in_chain_s"] - w_c) / w_c), flush=True)
        print("  D  %8.3f ms" % (dec["D_in_chain_s"] * 1e3)
              + ("   isolated %8.3f ms  -> composition %+.1f%%"
                 % (dec["D_isolated_s"] * 1e3,
                    100 * (dec["D_in_chain_s"] - dec["D_isolated_s"]) / dec["D_isolated_s"])
                 if dec.get("D_isolated_s") else ""), flush=True)
        c_flop = 2 * 2 * m * npix * no
        print("  C reaches %.1f TFLOP/s of a %.1f roof (%.0f%%)"
              % (c_flop / dec["C_in_chain_s"] / 1e12, MM_ROOF / 1e12,
                 100 * c_flop / dec["C_in_chain_s"] / MM_ROOF), flush=True)

        # --- the iteration ---------------------------------------------------------------------
        nblock = a.n_part / B
        # the floor is written against the PADDED translation count, because that is the work the
        # arm actually issues; the 7 dead translations are a real cost, not an accounting artefact.
        fl = floor_terms(a.n_part, no, ntp, npix, a.n_sig, B)
        meas = {"A_fft_s": a.n_part / FFT_IMG_S,
                "B_proj_s": no / PROJ_SLICE_S,
                "S_shift_s": nblock * dec["S_in_chain_s"],
                "C_compare_s": nblock * dec["C_in_chain_s"],
                "D_weights_s": nblock * dec["D_in_chain_s"],
                "E_bproj_s": a.n_part * a.n_sig / BPROJ_SLICE_S}
        tot, ftot = sum(meas.values()), sum(fl.values())
        res.update({"n_block": nblock, "iteration_floor_s": fl, "iteration_measured_s": meas,
                    "iteration_total_s": tot, "iteration_floor_total_s": ftot,
                    "pct_of_floor": 100 * ftot / tot})
        print(f"\nONE E-STEP ITERATION, N_p={a.n_part}, N_sig={a.n_sig}, {nblock:.0f} blocks of {B}:",
              flush=True)
        print("  %-14s %10s %10s %8s   %s" % ("term", "floor ms", "arm ms", "% iter", "provenance"),
              flush=True)
        prov = {"A_fft_s": "rate MEASURED, fft pass", "B_proj_s": "rate MEASURED, proj pass",
                "S_shift_s": "MEASURED here, in chain", "C_compare_s": "MEASURED here, in chain",
                "D_weights_s": "MEASURED here, in chain", "E_bproj_s": "rate MEASURED, bproj pass"}
        for k in ("A_fft_s", "B_proj_s", "S_shift_s", "C_compare_s", "D_weights_s", "E_bproj_s"):
            print("  %-14s %10.1f %10.1f %7.1f%%   %s"
                  % (k[:-2], fl[k] * 1e3, meas[k] * 1e3, 100 * meas[k] / tot, prov[k]), flush=True)
        print("  %-14s %10.1f %10.1f %7.1f%%   -> %.1f%% of the composed floor"
              % ("TOTAL", ftot * 1e3, tot * 1e3, 100.0, 100 * ftot / tot), flush=True)

        suffix = f"_{a.tag}" if a.tag else ""
        out = HERE / f"estep_e2e_box{a.box}_B{B}_no{no}{suffix}.json"
        out.write_text(json.dumps(res, indent=1))
        print("wrote", out, flush=True)
    finally:
        ttnn.close_device(dev)


if __name__ == "__main__":
    main()
