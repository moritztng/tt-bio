#!/usr/bin/env python3
"""C12 pass 23: the layout/data-movement block, the largest unowned item after the reblocks.

`c12-profiled-fold` measured all 19 device ops in the fold. Ten of them compute none of the
model's arithmetic -- they move or re-lay-out bytes. Nobody has priced them, and together they are
larger than the campaign's remaining gap to 12.5 s. This screens them, sizes the one shape worth a
row, and states what the screen cannot settle.

SOURCES
  composed.json   whole-fold per-device-op seconds, six units weighted by the fold's own integer
                  call counts, held+during-sampled 1350 MHz.
                  wk/c12-profiled-fold @ a63d6d8e3, perf/c12_profiled_fold/runs/composed.json
  ops CSVs        per-op rows (OP CODE, DEVICE KERNEL DURATION, CORE COUNT, operand shapes) for
                  the PairformerLayer and MSALayer legs, same branch,
                  perf/c12_profiled_fold/runs/{pfl_prof2,msal_prof}/ops_perf_results.csv.gz
  roofs           qb2 card 3, one interleaved session, forced+sampled 1350 MHz, from
                  c12-genericop-rate: 435.73 GB/s at 2R+1W, 393.31 GB/s at 1R+1W.

METHOD NOTE, and it is load-bearing. Whole-fold totals come ONLY from composed.json. The CSV is
used ONLY for the shape split INSIDE one op. Deriving totals from CSV time-fractions does not work:
the capture window contains once-per-capture setup alongside per-rep work, so a fraction-based
extrapolation over-attributes exactly the ops with tiny row counts -- on the PairformerLayer leg it
gives Untilize 0.1706 s, Tilize 0.1471 s and Embeddings 0.2373 s against composed.json's whole-fold
0.05904, 0.05358 and 0.05713 s, i.e. 3-4x too high at n = 8, 8 and 5 rows. Transpose (n = 121) and
ReshapeView (n = 14) do reproduce. So per-op totals are taken from composed.json and the CSV is
trusted only for relative weights within a single op code.

A transpose's compulsory traffic is read-once + write-once, so the 1R+1W roof is the right one.
c12-genericop-rate's correction applies: `bw_add8192` is the 2R+1W roof, not "the" DRAM roof.

Run: python3 layout_screen.py
"""

DEVICE_S = 13.2090
FOLD_S = 14.8810
GAP_TO_125 = 0.6040          # pass 23 book: 2.3810 needed - 1.7770 absurd best case

BW_1R1W_GBS = 393.31
BW_2R1W_GBS = 435.73

# composed.json /per_device_op_s -- the ten ops that compute no model arithmetic
LAYOUT_S = {
    "TransposeDeviceOperation":     0.51745,
    "SliceDeviceOperation":         0.19167,
    "ReshapeViewDeviceOperation":   0.15706,
    "PadDeviceOperation":           0.08641,
    "PermuteDeviceOperation":       0.08226,
    "ConcatDeviceOperation":        0.07479,
    "UntilizeDeviceOperation":      0.05904,
    "TilizeDeviceOperation":        0.05358,
    "CopyDeviceOperation":          0.02429,
    "NLPConcatHeadsDeviceOperation":0.01309,
}
# borderline: a head split is layout, but it is an attention-specific op so it is reported apart
NLP_CREATE_HEADS_S = 0.30067
REBLOCK_S = 1.0000           # inside generic_op, already opened as c12-reblock-delete

# Transpose shape split, pfl_prof2 leg, relative weights only (see METHOD NOTE).
# shape, rows in leg, us/call, share of the leg's Transpose time
TRANSPOSE_SHAPES = [
    ((1, 512, 1024, 512),  4, 2865.68, 0.1712),
    ((1, 512,  512, 128), 26,  733.78, 0.2850),
    ((1,  64,   32, 512), 32,   11.00, 0.0053),
    ((1,  64,  512,  32), 32,   10.95, 0.0052),
    ((1,  16,  512,  32), 18,    3.64, 0.0010),
    ((1,   1,  384, 512),  9,    2.94, 0.0004),
]
PFL_TRANSPOSE_S = 0.4681     # leg-derived, reproduces composed.json's 0.51745 whole-fold


def main():
    print("THE LAYOUT BLOCK: device ops that compute none of the model's arithmetic")
    tot = 0.0
    for k, v in sorted(LAYOUT_S.items(), key=lambda kv: -kv[1]):
        tot += v
        print(f"  {k:32s} {v:7.5f} s  {100*v/DEVICE_S:5.2f} % of device")
    print(f"  {'ten ops':32s} {tot:7.5f} s  {100*tot/DEVICE_S:5.2f} % of device, "
          f"{100*tot/FOLD_S:5.2f} % of fold")
    print(f"  {'+ NlpCreateHeads (head split)':32s} {NLP_CREATE_HEADS_S:7.5f} s  -> "
          f"{tot+NLP_CREATE_HEADS_S:.5f} s, {100*(tot+NLP_CREATE_HEADS_S)/DEVICE_S:.2f} % of device")
    print(f"  {'+ reblocks inside generic_op':32s} {REBLOCK_S:7.5f} s  -> "
          f"{tot+NLP_CREATE_HEADS_S+REBLOCK_S:.5f} s, "
          f"{100*(tot+NLP_CREATE_HEADS_S+REBLOCK_S)/DEVICE_S:.2f} % of device")
    print(f"\n  Only the {REBLOCK_S:.4f} s of reblocks has a row (c12-reblock-delete).")
    print(f"  UNOWNED: {tot:.4f} s (ten ops) to {tot+NLP_CREATE_HEADS_S:.4f} s (with the head split),")
    print(f"  against a remaining gap to 12.5 s of {GAP_TO_125:.4f} s -- "
          f"{tot/GAP_TO_125:.1f}x to {(tot+NLP_CREATE_HEADS_S)/GAP_TO_125:.1f}x the gap.")

    print(f"\nTRANSPOSE, the largest of them, by shape (PairformerLayer leg, {PFL_TRANSPOSE_S} s)")
    print(f"  {'shape':22s} {'rows':>5s} {'us/call':>9s} {'s of fold':>10s} {'MB 1R+1W':>9s} "
          f"{'roof ms':>8s} {'% of roof':>9s}")
    for sh, rows, us, s_fold in TRANSPOSE_SHAPES:
        n = sh[0]*sh[1]*sh[2]*sh[3]
        mb = n * 2 * 2 / 1e6                       # bf16, read once + write once
        roof_ms = mb / BW_1R1W_GBS                 # MB / (GB/s) = ms
        pct = 100 * roof_ms / (us/1000)
        print(f"  {'x'.join(map(str,sh)):22s} {rows:5d} {us:9.2f} {s_fold:10.4f} {mb:9.1f} "
              f"{roof_ms:8.3f} {pct:8.1f} %")
    top2 = sum(s for _,_,_,s in TRANSPOSE_SHAPES[:2])
    print(f"  two shapes are {top2:.4f} of {PFL_TRANSPOSE_S:.4f} s = "
          f"{100*top2/PFL_TRANSPOSE_S:.1f} % of the leg's Transpose time")

    print(f"\nWHAT THE SCREEN FINDS")
    big = TRANSPOSE_SHAPES[0]; small = TRANSPOSE_SHAPES[1]
    for sh, rows, us, s_fold in (big, small):
        n = sh[0]*sh[1]*sh[2]*sh[3]; mb = n*2*2/1e6
        roof_ms = mb / BW_1R1W_GBS; pct = 100*roof_ms/(us/1000)
        print(f"  {'x'.join(map(str,sh)):22s} {pct:5.1f} % of the 1R+1W roof", end="")
        if pct > 85:
            print(f"  -> AT THE ROOF, no rate headroom, deletion only")
        else:
            gain = s_fold * (1 - pct/100)
            print(f"  -> {100/pct:.1f}x above roof, so up to {gain:.4f} s if it")
            print(f"  {'':22s}        reached the rate its sibling shape demonstrably reaches")
    print(f"\n  The pair is the evidence: one transpose on this part reaches 95 % of the 1R+1W roof,")
    print(f"  which proves a transpose CAN stream at roof here, so the other sitting at 46 % is a")
    print(f"  real question rather than an assumed one.")
    print(f"\nWHAT THE SCREEN CANNOT SETTLE, and why this is a screen and not a prize")
    print(f"  1. A transpose's achievable rate is ACCESS-PATTERN dependent, and the two shapes swap")
    print(f"     different axes, so they are not interchangeable. The 46 % may be intrinsic.")
    print(f"  2. Call counts here are leg rows, not fold call counts. A row must read them from an")
    print(f"     executed graph -- a census/label count is not a call count in this campaign.")
    print(f"  3. Whether any of these transposes is CONTRACTUALLY REQUIRED by the layout its")
    print(f"     consumer demands is a source question nobody has asked. A layout op that cannot be")
    print(f"     removed is not a lever however far above roof it sits.")
    print(f"  So: a screen row, sized ~0.15 s on the one shape, NOT a build row, and it ranks below")
    print(f"  c12-reblock-delete (1.0000 s) which is the same mechanism -- deleting a DRAM pass.")


if __name__ == "__main__":
    main()
