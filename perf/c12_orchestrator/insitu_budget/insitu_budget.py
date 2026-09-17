#!/usr/bin/env python3
"""C12 pass 22: the budget re-based on MEASURED in-situ device time.

Passes 8-16 built every reachability verdict on a modelled device floor
(sum over ops of calls * max(t_traffic, t_arith)) subtracted from the fold.
Two rows have since measured the thing that floor was estimating:

  c12-profiled-fold   per-device-op seconds inside the fold, six profiled units
                      weighted by the fold's own integer call counts, held and
                      during-sampled 1350 MHz.  perf/c12_profiled_fold/runs/composed.json
                      on wk/c12-profiled-fold @ a63d6d8e3.
  c12-genericop-rate  the six generic_op sites against roofs measured on the
                      same part in the same session, with arithmetic intensity
                      checked against machine balance.
                      perf/c12_genop_rate/headroom.json on wk/c12-genericop-rate @ 4ed84475d.

Both numerators are transcribed from those artifacts, not from their DONE prose
(the prose nets the confidence-head bound off the remainder in one place and not
in another; this script carries both corners).

Run: python3 insitu_budget.py
"""

FOLD_S = 14.881          # c10-bare-baseline median, pinned during-sampled 1350 MHz
CLOCK_MHZ = 1350

# --- c12-profiled-fold, composed.json -------------------------------------
DEVICE_S = 13.2090       # /device_total_s
COVERAGE_PCT = 88.76     # /coverage_pct
CONF_HEAD_BOUND_S = 0.0231   # /bounded_unmeasured_s/ConfidenceHeadsDevice
REMAINDER_S = 1.672     # /remainder_s  (fold - device, confidence head still inside)

PER_DEVICE_OP_S = {           # /per_device_op_s, descending
    "MatmulDeviceOperation":   3.97049,
    "GenericOpDeviceOperation": 3.38300,
    "BinaryNgDeviceOperation":  2.38995,
    "LayerNormDeviceOperation": 1.38080,
    "TransposeDeviceOperation": 0.51745,
    "SDPAOperation":            0.43348,
    "NlpCreateHeadsDeviceOperation": 0.30067,
}
PER_DEVICE_OP_TAIL_S = 0.83300   # twelve smaller ops

PER_CLASS_S = {               # /per_class_s, the python-call-site split
    "linear": 3.39915, "generic_op": 3.38300, "multiply_": 1.49246,
    "layer_norm": 1.38080, "add_": 0.63476, "matmul": 0.57134,
}

UNITS = [   # /units -- unit, calls, device_ms_per_call, s_per_fold, programs_per_call
    ("PairformerLayer",        264, 30.65493, 8.0929, 137),
    ("DiffusionModule",        200, 20.33434, 4.0669, 1096),
    ("MSALayer",                16, 59.17096, 0.9467, 227),
    ("PairAssemblyDevice",       2, 26.58147, 0.0532, 27),
    ("RelPosGather",             2, 13.99937, 0.0280, 7),
    ("PairConditioningDevice",   1, 21.28959, 0.0213, 19),
]

# --- c12-genericop-rate, headroom.json ------------------------------------
# site, calls, in-situ sec, floor sec, FLOP/byte, pct of its own measured roof
GENOP_SITES = [
    ("trimul_in",     560, 0.839204, 0.517756, 106.61, 61.94),
    ("reblock_gated",1120, 0.722529, 0.573341,   0.00, 79.39),
    ("triatt_sdpa",   560, 0.706661, 0.347695, 254.01, 49.19),
    ("triatt_in",     560, 0.601210, 0.453037, 103.57, 75.37),
    ("reblock_back",  560, 0.277499, 0.191198,   0.00, 69.10),
    ("triatt_out",    560, 0.227183, 0.086293, 127.94, 37.99),
]
MACHINE_BALANCE_FPB = 260.9      # 435.7261 GB/s (2R+1W) vs 113.6833 TFLOP/s (cube4096 HiFi4)
GENOP_TOTAL_S = 3.374286
GENOP_FLOOR_S = 2.169320

# --- the campaign's named lever book, each with its numerator's provenance --
# name, fold seconds, provenance, status
# The `matmul` class is entered ONCE, at its measured in-situ cap, rather than as two overlapping
# leads. `c12-matmul-key-attribution` priced the whole class in situ at 0.6808 s against 1.4794 s
# booked and put total class headroom at <= 0.1352 s (PRICED.txt, TOTAL "argued" column). Both the
# 64-of-110 core pin and the cross-family arm are levers INSIDE that class, so adding them to it
# would double-count against a measured cap.
#
# The 0.2150 s core-pin figure handed over by `c12-genericop-rate` is VOID and is not used. The pin
# is real -- the armed capture reads CORE COUNT 64 on the trimul key -- but that key sits at 0.62x
# machine balance, i.e. on the TRAFFIC side, so a 110/64 = 1.72x occupancy multiplier does not apply
# to its rate. PRICED.txt computes what it would demand: 384.7 GB/s, which is 1.36x the best rate
# any real fold shape reached in that session. Its own line reads "its prize: envelope-style
# 0.7794 s, core-scaled 0.0552 s, argued 0.0584 s". This is the same error `c12-genericop-rate`
# named as its own headline lesson (a resource gap is not headroom until you check which resource
# binds), made on the lever it handed over, in the document that warned about it.
LEVERS = [
    ("silu",          0.2843, "executed-graph",  "GO op-level, accuracy clean, fold owed"),
    ("cond-hoist",    0.2415, "block-level A/B", "GO block-level, fold owed"),
    ("reblock-delete",1.0000, "in-situ measured","PRICED, UNBUILT, needs source build + accuracy"),
    ("matmul-class",  0.1352, "in-situ cap",     "<= this for ALL matmul levers; 64/110 pin is "
                                                 "0.0584 s of it, not 0.2150 s"),
    ("kblock",        0.0350, "production A/B",  "concluded BELOW its own kill criterion, off"),
    ("cross-family",  0.0810, "re-derived",      "row CLOSED on opportunity cost"),
    ("fused-eltwise", 0.1321, "in-situ",         "ACCURACY FAILED, default-off"),
]
BANKABLE = {"silu", "cond-hoist"}          # GO + accuracy clean, only a fold owed
BUILDABLE = {"reblock-delete", "matmul-class", "kblock", "cross-family"}

F_FITTED_S, F_ERR_S = 3.9830, 0.1181       # c10-fixed-cost, four pinned clock arms

TARGETS = [12.5, 10.0]


def line(c="-", n=78):
    print(c * n)


def main():
    print(f"C12 pass 22 -- budget re-based on measured in-situ device time")
    print(f"fold {FOLD_S:.4f} s at a held during-sampled {CLOCK_MHZ} MHz")
    line("=")

    # 1. the measured split, both corners of the remainder
    host_hi = REMAINDER_S
    host_lo = REMAINDER_S - CONF_HEAD_BOUND_S
    print("MEASURED SPLIT (c12-profiled-fold, six units, integer call weights)")
    print(f"  device, profiled            {DEVICE_S:8.4f} s   {100*DEVICE_S/FOLD_S:5.2f} %  "
          f"coverage {COVERAGE_PCT:.2f} %")
    print(f"  ConfidenceHeadsDevice       <= {CONF_HEAD_BOUND_S:5.4f} s   bounded, not profiled")
    print(f"  non-device remainder        {host_lo:8.4f} - {host_hi:.4f} s   "
          f"{100*host_lo/FOLD_S:5.2f} - {100*host_hi/FOLD_S:.2f} %")
    chk = DEVICE_S + host_hi
    print(f"  check  device + remainder = {chk:.4f} s vs fold {FOLD_S:.4f} s  "
          f"(delta {chk-FOLD_S:+.4f})")
    line()

    # 2. F against the measured host room -- Axis A
    print("AXIS A: F AGAINST THE MEASURED NON-DEVICE TERM")
    print(f"  F, fitted clock-immune      {F_FITTED_S:8.4f} +/- {F_ERR_S:.4f} s "
          f"(c10-fixed-cost, 4 pinned arms)")
    print(f"  measured non-device room    {host_lo:8.4f} - {host_hi:.4f} s")
    print(f"  F exceeds it by             {F_FITTED_S-host_hi:8.4f} - {F_FITTED_S-host_lo:.4f} s"
          f"   -> that much of F is clock-immune DEVICE cost, not host code")
    host_only = FOLD_S - host_hi
    print(f"  fold with 100 % of non-device time deleted = {host_only:.4f} s")
    for t in TARGETS:
        verdict = "REACHES" if host_only <= t else f"MISSES by {host_only-t:.4f} s"
        print(f"    vs {t:5.1f} s target: {verdict}")
    print("  => host work cannot reach either target even if deleted entirely.")
    line()

    # 3. where the device seconds are
    print(f"DEVICE SECONDS BY DEVICE OP (measured in situ, {DEVICE_S:.4f} s total)")
    tot = 0.0
    for k, v in PER_DEVICE_OP_S.items():
        tot += v
        print(f"  {k:34s} {v:7.4f} s  {100*v/DEVICE_S:5.2f} %  {100*v/FOLD_S:5.2f} % of fold")
    print(f"  {'twelve smaller':34s} {PER_DEVICE_OP_TAIL_S:7.4f} s  "
          f"{100*PER_DEVICE_OP_TAIL_S/DEVICE_S:5.2f} %")
    tot += PER_DEVICE_OP_TAIL_S
    print(f"  {'sum':34s} {tot:7.4f} s  (vs {DEVICE_S:.4f}, delta {tot-DEVICE_S:+.4f})")
    line()

    print(f"DEVICE SECONDS BY UNIT")
    for name, calls, mspc, spf, ppc in UNITS:
        print(f"  {name:24s} {calls:4d} x {mspc:8.4f} ms = {spf:7.4f} s  "
              f"{100*spf/DEVICE_S:5.2f} %  {calls*ppc:7d} programs")
    line()

    # 4. generic_op: the arithmetic-bound premise, tested against machine balance
    print("GENERIC_OP: ARITHMETIC INTENSITY vs MEASURED MACHINE BALANCE")
    print(f"  machine balance {MACHINE_BALANCE_FPB:.1f} FLOP/byte "
          f"(435.73 GB/s 2R+1W, 113.68 TFLOP/s cube4096 HiFi4, same part same session)")
    zero_flop = 0.0
    for site, calls, sec, floor, fpb, pct in GENOP_SITES:
        bound = "arith" if fpb > MACHINE_BALANCE_FPB else "TRAFFIC"
        if fpb == 0.0:
            zero_flop += sec
        print(f"  {site:15s} {calls:5d} calls  {sec:7.4f} s  floor {floor:7.4f}  "
              f"{fpb:7.2f} FLOP/b  {pct:5.2f} % of own roof  {bound}")
    n_arith = sum(1 for s in GENOP_SITES if s[4] > MACHINE_BALANCE_FPB)
    print(f"  arithmetic-bound sites: {n_arith} of {len(GENOP_SITES)}")
    print(f"  in situ {GENOP_TOTAL_S:.4f} s, floor {GENOP_FLOOR_S:.4f} s, "
          f"gap-to-own-roofs {GENOP_TOTAL_S-GENOP_FLOOR_S:.4f} s")
    print(f"  ZERO-FLOP (reblock) time  {zero_flop:.4f} s = "
          f"{100*zero_flop/GENOP_TOTAL_S:.1f} % of the op, {100*zero_flop/DEVICE_S:.1f} % of device")
    print("  => passes 10-11 priced this op arithmetic-bound and specified a 1.50-2.17x")
    print("     rate ask.  Zero of six sites is arithmetic-bound.  That spec is void.")
    line()

    # 5. the target arithmetic, all of it now out of device time
    print("WHAT EACH TARGET NEEDS, with host capped at the measured remainder")
    for t in TARGETS:
        need = FOLD_S - t
        dev_needed = DEVICE_S - max(0.0, need - host_hi)
        print(f"  {t:5.1f} s: needs {need:.4f} s total; host can supply at most {host_hi:.4f} s, "
              f"so device must fall")
        print(f"          {DEVICE_S:.4f} -> {dev_needed:.4f} s "
              f"({100*(DEVICE_S-dev_needed)/DEVICE_S:.2f} % of device) in the best case for host")
    line()

    # 6. the book
    print("THE NAMED LEVER BOOK, with each numerator's provenance")
    for name, s, prov, status in LEVERS:
        print(f"  {name:16s} {s:7.4f} s  {prov:16s} {status}")
    bank = sum(s for n, s, _, _ in LEVERS if n in BANKABLE)
    build = sum(s for n, s, _, _ in LEVERS if n in BUILDABLE)
    print(f"\n  bankable (GO, accuracy clean, fold owed)      {bank:7.4f} s")
    print(f"  + every unbuilt named lever at FULL priced value {build:7.4f} s")
    print(f"  = absurd best case                             {bank+build:7.4f} s")
    for t in TARGETS:
        need = FOLD_S - t
        print(f"    vs {t:5.1f} s (needs {need:.4f} s): bankable {100*bank/need:5.1f} %, "
              f"best case {100*(bank+build)/need:5.1f} %, "
              f"short {need-bank-build:+.4f} s")
    print(f"\n  and with 100 % of non-device time deleted on top of the absurd best case:")
    for t in TARGETS:
        landed = FOLD_S - bank - build - host_hi
        print(f"    fold {landed:.4f} s vs {t:5.1f} s -> "
              f"{'REACHES' if landed <= t else f'MISSES by {landed-t:.4f} s'}")
    line("=")
    print("VERDICT: 12.5 s turns entirely on reblock-delete landing near its 1.0000 s price.")
    print("         10.0 s misses even when every named lever lands in full AND all")
    print("         non-device time is deleted -- the fourth independent derivation.")


if __name__ == "__main__":
    main()
