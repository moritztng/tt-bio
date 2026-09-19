#!/usr/bin/env python3
"""Pair-track MAC counts per Pairformer block, and the measured time beside them.

Host-only. Every structural constant here is read off the checkpoint or off the
constructor call that builds the module, never off a paper. Doubling c_z is 2x on the
N^3 contractions and 4x on the N^2 c^2 projections, and at N = 512 the projections are
half of protenix's block, so "2x wider, so 2x the work" understates the ratio by 48 %.
"""
N = 512


def block(c):
    """MACs in one Pairformer block's pair track at pair width `c`."""
    U3, U2 = N ** 3 * c, N ** 2 * c * c
    return {
        "trimul_contract": 2 * U3,      # a[i,k,c].b[j,k,c], one per trimul, two trimuls
        "trimul_proj": 12 * U2,         # a_p a_g b_p b_g g z, six each, two trimuls
        "triatt_contract": 4 * U3,      # QK^T and AV are N^3 (h d) = N^3 c each, two attentions
        "triatt_proj": 10 * U2,         # q k v o g, five each, two attentions
        "transition": 3 * (N ** 2) * c * (4 * c),   # swiglu: two in-projections plus one out
    }


PVX, B2 = block(256), block(128)
print(f"{'term':<20}{'boltz2 c=128':>16}{'protenix c=256':>18}{'ratio':>8}")
for k in PVX:
    print(f"{k:<20}{B2[k]:16.4e}{PVX[k]:18.4e}{PVX[k]/B2[k]:8.2f}x")
tb, tp = sum(B2.values()), sum(PVX.values())
print(f"{'BLOCK TOTAL':<20}{tb:16.4e}{tp:18.4e}{tp/tb:8.3f}x")
print(f"\nblock executions  protenix 48x10 = {48*10}   boltz2 64x4 = {64*4}"
      f"   ratio {480/256:.3f}x")
print(f"TRUNK PAIRFORMER WORK RATIO = {tp/tb:.3f} x {480/256:.3f} = {tp/tb*480/256:.3f}x")

# MEASURED, perf/pvx_ps/p1.json deep arm, self seconds under fold/trunk_cond/trunk/pairformer
MEAS = {"trimul": 11.218, "triatt": 9.631, "transition": 7.338,
        "attn_pair_bias": 1.503, "pf_layer_self": 2.742}
FLOP = {"trimul": PVX["trimul_contract"] + PVX["trimul_proj"],
        "triatt": PVX["triatt_contract"] + PVX["triatt_proj"],
        "transition": PVX["transition"], "attn_pair_bias": 0.0, "pf_layer_self": 0.0}
ts, fs = sum(MEAS.values()), sum(FLOP.values())
print(f"\n{'body':<16}{'self s':>9}{'time %':>9}{'flop %':>9}{'t%/f%':>8}")
for k in MEAS:
    f = 100 * FLOP[k] / fs
    print(f"{k:<16}{MEAS[k]:9.3f}{100 * MEAS[k] / ts:9.1f}{f:9.1f}"
          f"{(100 * MEAS[k] / ts) / f if f else float('inf'):8.2f}")
print(f"{'TOTAL':<16}{ts:9.3f}")
zero = MEAS["attn_pair_bias"] + MEAS["pf_layer_self"]
print(f"\ncarrying ~no pair-track arithmetic: {zero:.3f} s = {100 * zero / ts:.1f} % of the "
      f"Pairformer")

# The cross-model comparison, deflated to the uninstrumented fold
CENSUS_FOLD, CLEAN_FOLD = 55.205, 52.077
PF_CENSUS = 32.455
pf_clean = PF_CENSUS * CLEAN_FOLD / CENSUS_FOLD
b2_frac = 8.0929 / 14.881          # MEASURED (prior): PairformerLayer of the boltz2 fold
b2_p150a = 17.39                   # MEASURED (prior): the same fold on a p150a at 1350 MHz
print(f"\nprotenix trunk Pairformer, deflated to the clean fold: "
      f"{PF_CENSUS:.3f} x {CLEAN_FOLD}/{CENSUS_FOLD} = {pf_clean:.2f} s")
print(f"boltz2 trunk Pairformer on a p150a: {b2_frac:.4f} x {b2_p150a} = "
      f"{b2_frac * b2_p150a:.2f} s")
print(f"measured Pairformer ratio {pf_clean / (b2_frac * b2_p150a):.3f}x against a work ratio "
      f"of {tp / tb * 480 / 256:.3f}x")
print(f"-> protenix executes the SHARED Pairformer "
      f"{(tp / tb * 480 / 256) / (pf_clean / (b2_frac * b2_p150a)):.3f}x more efficiently per MAC")
print(f"whole fold: {CLEAN_FOLD} / {b2_p150a} = {CLEAN_FOLD / b2_p150a:.3f}x")
