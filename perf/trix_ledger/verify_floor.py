#!/usr/bin/env python3
"""Independently re-derive TRIX's compulsory floor from the mathematics. CPU only, no device.

`trix-floor` (state/trix-floor.md, VERDICT: GO) replaced the campaign's 6.61 ms "arithmetic floor"
with 2.648 ms/call compulsory, built on 274.877 GFLOP and 1073.7 MB at 512 aa / c_z=256. Every score
in the campaign is now divided by that, so the orchestrator re-derives it here from the definition
of the operation rather than reproducing the row's own script. Same answer from a different route is
the check; a different answer is a finding.

Triangle multiplication, one call, batch 1, bf16, N tokens, c_z channels, hidden == c_z (forced by
the reference: p_in and g_in are dim -> 2*dim and the chunk halves that back to c_z).
"""

N, CZ, BYTES = 512, 256, 2          # 512 aa, c_z = 256, bf16
C = CZ                              # hidden == c_z


def gflop():
    """Compulsory multiply-accumulates x2, by leg."""
    legs = {
        # p_in and g_in: each [N,N,c_z] @ [c_z, 2*c_z]; the chunk halves 2*c_z back to c_z
        "in_proj (p and g)": 2 * (2 * N * N * CZ * (2 * CZ)),
        # out[i,j,c] = sum_k a[i,k,c] * b[j,k,c]  ->  N*N*N*c MACs
        "contraction":       2 * N * N * N * C,
        # [N,N,c] @ [c, c_z]
        "out_proj":          2 * N * N * C * CZ,
        # output gate, [N,N,c_z] @ [c_z, c_z]
        "out_gate":          2 * N * N * CZ * CZ,
    }
    return legs, sum(legs.values()) / 1e9


def z_bytes():
    """One pair tensor in bf16."""
    return N * N * CZ * BYTES


def main():
    legs, total = gflop()
    print(f"N={N}  c_z={CZ}  hidden={C}  dtype=bf16\n")
    print("COMPULSORY ARITHMETIC")
    for k, v in legs.items():
        print(f"  {k:22s} {v/1e9:8.2f} GFLOP")
    print(f"  {'TOTAL':22s} {total:8.3f} GFLOP     trix-floor: 274.877  ->  "
          f"{'MATCH' if abs(total - 274.877) < 0.01 else 'MISMATCH'}")

    Z = z_bytes()
    passes = 8                       # trix-floor's compulsory dataflow: 8 Z-sized passes
    traffic = passes * Z
    print(f"\nCOMPULSORY TRAFFIC")
    print(f"  Z (one pair tensor)    {Z/1e6:8.2f} MB")
    print(f"  {passes} Z-sized passes      {traffic/1e6:8.1f} MB     trix-floor: 1073.7  ->  "
          f"{'MATCH' if abs(traffic/1e6 - 1073.7) < 0.1 else 'MISMATCH'}")

    # Which side binds, at trix-floor's own measured roofs (pinned 1350 MHz, shipped kernel config)
    COMPUTE_ROOF, DRAM_COMBINED = 123.65e12, 410.3e9
    t_flops = total * 1e9 / COMPUTE_ROOF
    t_bytes = traffic / DRAM_COMBINED
    print(f"\nWHICH SIDE BINDS (roofs: {COMPUTE_ROOF/1e12} TFLOP/s, {DRAM_COMBINED/1e9} GB/s)")
    print(f"  arithmetic             {t_flops*1e3:8.3f} ms")
    print(f"  traffic                {t_bytes*1e3:8.3f} ms   <-- binds"
          if t_bytes > t_flops else "  traffic binds: no")
    print(f"  => floor is {'BYTE' if t_bytes > t_flops else 'COMPUTE'}-bound, "
          f"{max(t_flops, t_bytes)*1e3:.3f} ms at roof")
    print(f"  trix-floor quotes 2.648 ms, i.e. {traffic/2.648e-3/1e9:.1f} GB/s achieved = "
          f"{100*traffic/2.648e-3/DRAM_COMBINED:.1f} % of the combined roof")

    print(f"\nWHAT THE RETIRED 6.61 ms WAS")
    print(f"  6.61 / {max(t_flops, t_bytes)*1e3:.3f} = {6.61/(max(t_flops,t_bytes)*1e3):.2f}x too high")
    print(f"  trix-floor attributes that to pricing the arithmetic at our own kernels' rates "
          f"(2.16x)\n  and reading those rates at a governor-set clock (a further 1.16x): "
          f"2.16 * 1.16 = {2.16*1.16:.2f}x")


if __name__ == "__main__" and not (set(__import__("sys").argv) & {"--carryout", "--ceiling", "--floors", "--progress"}):
    main()


# ---------------------------------------------------------------------------
# Cross-row check: what survives a stale denominator, and what does not.
#
# `trix-transaction` carried its measured op-level win out to the module and the fold. The op number
# is a direct measurement; the module RATIO turned out to be divided by the retired standalone wall.
# This reproduces the check so the correction is auditable rather than a one-off shell computation.
# Run: python3 verify_floor.py --carryout
# ---------------------------------------------------------------------------

BASE_MS, BANK8_MS, NOREAD_MS = 1.4490, 1.1323, 1.1129   # shipped-kernel ablation, 1350 MHz
OPS_PER_MODULE_CALL = 2
IN_FOLD_FACTOR, CALLS = 0.906, 1208
MODULE_TODAY_MS, MODULE_RETIRED_MS = 12.330, 18.617     # trix-floor vs the retired standalone


def carryout():
    op_sav = BASE_MS - BANK8_MS
    mod_sav = op_sav * OPS_PER_MODULE_CALL
    fold_sav_s = mod_sav * IN_FOLD_FACTOR * CALLS / 1e3
    print(f"op ratio                {BASE_MS/BANK8_MS:.4f}x   (direct measurement, no denominator)")
    exposed = BASE_MS - NOREAD_MS            # what the DRAM read actually costs, exposed
    print(f"  ceiling: noread       {BASE_MS/NOREAD_MS:.4f}x   (read deleted entirely)")
    # Three ways to say "how much of the read side does the fix capture", and they differ.
    # The defensible one is TIME: of the milliseconds the read costs exposed, how many go away.
    # `core-coverage-ratio-not-recoverable-time` is the standing lesson -- a ratio of speedups is
    # not a fraction of a recoverable quantity, and it flatters.
    print(f"  read exposed          {exposed:.4f} ms; the fix removes {op_sav:.4f} ms = "
          f"{100*op_sav/exposed:.1f} % OF THE RECOVERABLE TIME")
    print(f"    (the row quoted 98 %, which is {BASE_MS/BANK8_MS:.4f}/{BASE_MS/NOREAD_MS:.4f}, a "
          f"ratio of speedups rather than a fraction of time;")
    print(f"     the excess-speedup form is a third number, "
          f"{100*(BASE_MS/BANK8_MS - 1)/(BASE_MS/NOREAD_MS - 1):.1f} %. Quote the time one.)")
    print(f"per-op saving           {op_sav:.4f} ms")
    print(f"per module call         {mod_sav:.4f} ms   ({OPS_PER_MODULE_CALL} ops per call)")
    print()
    print("MODULE RATIO -- inherits whichever wall you divide by:")
    print(f"  vs retired {MODULE_RETIRED_MS} ms  {MODULE_RETIRED_MS/(MODULE_RETIRED_MS-mod_sav):.4f}x"
          f"   <- what the row wrote (1.0352x)")
    print(f"  vs today's {MODULE_TODAY_MS} ms  {MODULE_TODAY_MS/(MODULE_TODAY_MS-mod_sav):.4f}x"
          f"   <- correct; the correction makes the lever BIGGER")
    print()
    print(f"FOLD SAVING             {fold_sav_s:.4f} s")
    print(f"  = {mod_sav:.4f} ms x {IN_FOLD_FACTOR} in-fold x {CALLS} calls")
    print("  uses NO module wall, so no staleness can reach it -- pre-register THIS, not a ratio.")
    print("  A fold RATIO needs a fold wall measured in the same session; the 71.9 s in circulation")
    print("  has not been re-measured on today's tree either.")


if __name__ == "__main__" and "--carryout" in __import__("sys").argv:
    carryout()


# ---------------------------------------------------------------------------
# The campaign's ceiling, on measured denominators.
#
# `trix-scaffold-attribute` measured the module IN-FOLD on today's tree and, with it, the fold wall
# itself -- the two denominators that had been stale. This turns "eliminate the trimul bottleneck"
# into a number. Run: python3 verify_floor.py --ceiling
# ---------------------------------------------------------------------------

MODULE_INFOLD_S, FOLD_S, TRIMUL_CALLS = 12.6881, 48.62, 1208   # 512 aa protenix-v2 cdk2x2_512
GATED_SHARE, GATED_CALLS = 0.1969, 2096                        # reblock_permute_gated
BANK_RATIO = BASE_MS / BANK8_MS                                # 1.2797x, measured on the shipped op


def ceiling():
    floor_s = 2.648e-3 * TRIMUL_CALLS
    print(f"module in-fold          {MODULE_INFOLD_S} s over {TRIMUL_CALLS} calls "
          f"= {MODULE_INFOLD_S/TRIMUL_CALLS*1e3:.3f} ms/call")
    print(f"trimul share of fold    {100*MODULE_INFOLD_S/FOLD_S:.2f} %   "
          f"(the campaign was framed at 30.939/78.24 = {100*30.939/78.24:.1f} %)")
    print()
    print("WHAT 'ELIMINATE THIS BOTTLENECK' IS WORTH, on measured denominators:")
    print(f"  trimul deleted entirely      {FOLD_S/(FOLD_S-MODULE_INFOLD_S):.4f}x on the fold")
    print(f"  trimul at its compulsory floor ({floor_s:.3f} s)"
          f"  {FOLD_S/(FOLD_S-MODULE_INFOLD_S+floor_s):.4f}x   <- the achievable ceiling")
    print("  NOTE the floor is the 512 aa / D=256 figure applied to all 1208 calls; ~160 run the")
    print("  narrow C=64 path with a SMALLER floor, so the true ceiling is marginally HIGHER.")
    print()
    print("BANK SPREAD re-grounded on the measured in-fold share (not a carried-out factor):")
    gated_s = GATED_SHARE * MODULE_INFOLD_S
    sav = gated_s * (1 - 1 / BANK_RATIO)
    print(f"  reblock_permute_gated        {gated_s:.4f} s in-fold over {GATED_CALLS} calls "
          f"= {gated_s/GATED_CALLS*1e3:.4f} ms/call")
    print(f"  in-fold factor for this op   {gated_s/GATED_CALLS*1e3/BASE_MS:.4f} "
          f"(the carry-out assumed 0.906)")
    print(f"  saving at the measured {BANK_RATIO:.4f}x  {sav:.4f} s "
          f"-> {FOLD_S/(FOLD_S-sav):.4f}x on the fold")
    print(f"  the carry-out said 0.693 s / 1.0097x: smaller absolute, LARGER ratio, because the")
    print(f"  fold is {FOLD_S} s and not the 71.9 s it divided by.")


if __name__ == "__main__" and "--ceiling" in __import__("sys").argv:
    ceiling()


# ---------------------------------------------------------------------------
# THREE floors, not one. `trix-floor` and `trix-radical` disagreed, and the disagreement is real
# and useful: they answer different questions and only one of them is "compulsory".
#
#   compulsory        what ANY correct implementation must do        -> 3Z, COMPUTE-bound
#   shipped-dataflow  the best TODAY's schedule could do at roof     -> 8Z, BYTE-bound
#   achievable        today's schedule at today's kernel class rates -> trix-floor's 5.715 ms
#
# Run: python3 verify_floor.py --floors
# ---------------------------------------------------------------------------

DRAM_COMBINED_ROOF = 410.3e9
ACHIEVABLE_MS = 5.715                      # trix-floor, today's dataflow at today's kernel rates


def floors():
    flops = 12 * N * N * CZ * CZ + 2 * N**3 * CZ      # trix-radical's closed form
    Z = z_bytes()
    t_arith = flops / 123.65e12 * 1e3
    print(f"arithmetic  12N^2D^2 + 2N^3D = {flops/1e9:.3f} GFLOP"
          f"   (leg-by-leg derivation agrees exactly)")
    print(f"Z = N^2 * D * 2 B = {Z/1e6:.3f} MB;  arithmetic at the measured roof = {t_arith:.3f} ms\n")
    for name, zn, note in (
        ("COMPULSORY       ", 3, "z read twice, answer written once. 2Z is unreachable: reading z "
                                 "once forces a, b\n                     and the output gate "
                                 "simultaneously live = 3Z residency against a 134.1 MB L1\n"
                                 "                     ceiling, and even bfp8_b is 213.9 MB -- so "
                                 "the bound does not depend on precision."),
        ("SHIPPED-DATAFLOW ", 8, "what today's schedule actually moves. trix-floor called this "
                                 "compulsory; it is the\n                     floor OF THIS "
                                 "DATAFLOW, which is a different and also useful question."),
    ):
        t_b = zn * Z / DRAM_COMBINED_ROOF * 1e3
        binds = "BYTES" if t_b > t_arith else "COMPUTE"
        print(f"{name} {zn}Z = {zn*Z/1e6:7.1f} MB -> {t_b:.3f} ms traffic; {binds}-bound; "
              f"floor {max(t_b, t_arith):.3f} ms")
        print(f"                     {note}\n")
    print(f"ACHIEVABLE        {ACHIEVABLE_MS:.3f} ms -- today's dataflow at today's kernel class rates\n")
    print(f"byte headroom at the compulsory floor: {t_arith/(3*Z/DRAM_COMBINED_ROOF*1e3):.2f}x")
    print("=> TODAY'S IMPLEMENTATION IS BYTE-LIMITED; A PERFECT ONE WOULD BE COMPUTE-LIMITED.")
    print("   The gap between them is exactly the 8Z -> 3Z traffic collapse.\n")
    for lbl, fl in (("compulsory 2.223", 2.223), ("shipped-dataflow 2.648", 2.648)):
        f_s = fl * 1e-3 * TRIMUL_CALLS
        print(f"  ceiling with the {lbl} ms floor: trimul at floor = {f_s:.3f} s "
              f"-> {FOLD_S/(FOLD_S-MODULE_INFOLD_S+f_s):.4f}x on the fold")


if __name__ == "__main__" and "--floors" in __import__("sys").argv:
    floors()


# ---------------------------------------------------------------------------
# What the campaign is actually worth, past and future, on the same fixture.
# Run: python3 verify_floor.py --progress
# ---------------------------------------------------------------------------

FOLD_AUG, TRIMUL_AUG = 78.24, 30.939      # trimul-bottleneck-rootcause, protenix-v2 512 aa
COMPULSORY_MS = 2.223                     # trix-radical, compute-bound


def progress():
    rest_aug, rest_now = FOLD_AUG - TRIMUL_AUG, FOLD_S - MODULE_INFOLD_S
    print("Same protenix-v2 cdk2x2_512 fixture, Aug 2026 against today:\n")
    print(f"  {'':16s}{'Aug':>10s}{'today':>10s}{'speedup':>10s}")
    for lbl, a, b in (("fold", FOLD_AUG, FOLD_S),
                      ("trimul", TRIMUL_AUG, MODULE_INFOLD_S),
                      ("everything else", rest_aug, rest_now)):
        print(f"  {lbl:16s}{a:9.2f}s{b:9.2f}s{a/b:9.2f}x")
    print(f"\n  trimul share of fold: {100*TRIMUL_AUG/FOLD_AUG:.1f} % -> {100*MODULE_INFOLD_S/FOLD_S:.1f} %")
    print(f"  trimul outpaced the rest of the fold by "
          f"{(TRIMUL_AUG/MODULE_INFOLD_S)/(rest_aug/rest_now):.2f}x")
    floor_s = COMPULSORY_MS * 1e-3 * TRIMUL_CALLS
    end = FOLD_S - MODULE_INFOLD_S + floor_s
    print(f"\n  REMAINING CEILING: trimul at its compulsory floor ({floor_s:.3f} s) takes the fold")
    print(f"  {FOLD_S:.2f} -> {end:.2f} s = {FOLD_S/end:.4f}x. Size every proposal against that.")


if __name__ == "__main__" and "--progress" in __import__("sys").argv:
    progress()
