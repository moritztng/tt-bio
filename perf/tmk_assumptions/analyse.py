#!/usr/bin/env python3
"""Turn rate_gap.json into the binding-roof table. CPU only, no device.

A TFLOP/s gap is headroom only if the shape's arithmetic intensity says it can be reached
(`roof-headroom-needs-arithmetic-intensity-not-just-tflops-gap`), so every call site is scored
against min(compute roof, its own traffic roof) and never against the square cube.

Two traffic roofs are quoted per site and neither is modelled: the SAME-MIX roof, measured with a
bandwidth op that moves the same read:write split, and the BEST roof, the fastest DRAM stream seen
anywhere in the session. The first is the realistic ceiling, the second is a deliberately generous
one, and the honest headroom is the band between them.
"""
import json, sys

d = json.load(open(sys.argv[1]))
A = d["arms"]
CAMPAIGN_ROOF = 123.65


def best(*names, k="serial_GBs"):
    v = [A[n][k] for n in names if n in A]
    return max(v) if v else None


cube = max(A["roof_cube4096"]["serial_TFLOPs"], A["roof_cube4096_AA"]["serial_TFLOPs"])
bw_2R1W = best("roof_dram_2R1W", "roof_dram_2R1W_AA")
bw_1R1W = best("roof_dram_1R1W", "roof_dram_clone")
bw_1R2W = best("roof_dram_1R2W")
bw_read = best("roof_dram_1R0W")
bw_best = max(x for x in (bw_2R1W, bw_1R1W, bw_1R2W, bw_read) if x)

print(f"IN-SESSION ROOFS  cube4096 {cube:.2f} TFLOP/s (campaign quotes {CAMPAIGN_ROOF})")
print(f"  DRAM  2R1W add {bw_2R1W:.1f} | 1R1W {bw_1R1W:.1f} | 1R2W concat {bw_1R2W:.1f} | "
      f"read-only {bw_read} GB/s  -> best {bw_best:.1f}")
print(f"  machine balance at the cube, against the best DRAM stream: "
      f"{cube * 1e12 / (bw_best * 1e9):.0f} FLOP/byte")
print()

SITES = [
    ("A_inproj_DRAM",    "in-projection  [1,1,262144,256]@[256,512]", bw_1R2W, "1R:2W"),
    ("B_contract_DRAM",  "contraction    [1,128,512,512] x same",     bw_2R1W, "2R:1W"),
    ("C_outproj_linear", "out-projection [1,512,512,256]@[256,256]",  bw_1R1W, "1R:1W"),
]
print(f"{'site':44s} {'TF/s':>6s} {'GB/s':>6s} {'AI':>6s} | "
      f"{'same-mix':>9s} {'%':>6s} | {'best-BW':>8s} {'%':>6s} | {'headroom band':>14s} | claimed")
print("-" * 132)
rows = {}
for key, label, bwmix, mix in SITES:
    a = A[key]
    ai = a["AI_flop_per_byte"]
    # A data-movement op is not a bandwidth roof: the 1R:2W concat arm reads BELOW what the
    # in-projection itself achieves, so it is refused as a roof rather than used as one.
    bwmix_eff = max(bwmix, a["serial_GBs"])
    t_mix = min(cube, ai * bwmix_eff / 1e3)
    t_best = min(cube, ai * bw_best / 1e3)
    r = a["serial_TFLOPs"]
    rows[key] = dict(rate=r, t_mix=t_mix, t_best=t_best, GFLOP=a["GFLOP"], ms=a["serial_ms"])
    if bwmix_eff > bwmix:
        print(f"  (same-mix roof for this site raised {bwmix:.1f} -> {bwmix_eff:.1f} GB/s: the "
              f"{mix} bandwidth arm is slower than the call site, so it is not a roof)")
    print(f"{label:44s} {r:6.2f} {a['serial_GBs']:6.1f} {ai:6.1f} | "
          f"{t_mix:9.2f} {100 * r / t_mix:5.1f}% | {t_best:8.2f} {100 * r / t_best:5.1f}% | "
          f"{t_mix / r:6.2f}x-{t_best / r:5.2f}x | {CAMPAIGN_ROOF / r:5.2f}x")
print()

# --- the module table, rebuilt at the roofs that actually bind ------------------
GF = {"A_inproj_DRAM": 137.438, "B_contract_DRAM": 68.719, "C_outproj_linear": 68.719}
at_campaign = sum(v / CAMPAIGN_ROOF for v in GF.values())
at_rate = sum(GF[k] / rows[k]["rate"] for k in GF)
at_mix = sum(GF[k] / rows[k]["t_mix"] for k in GF)
at_best = sum(GF[k] / rows[k]["t_best"] for k in GF)
print("THE MODULE'S 274.877 GFLOP, priced four ways (ms/call):")
print(f"  at the campaign's 123.65 square roof        {at_campaign:7.3f}   <- the 2.223 ms line")
print(f"  at each site's OWN best-BW roof             {at_best:7.3f}")
print(f"  at each site's OWN same-mix roof            {at_mix:7.3f}")
print(f"  at today's measured call-site rates         {at_rate:7.3f}")
print(f"  campaign 'kernel-rate deficit'              {at_rate - at_campaign:7.3f} ms")
print(f"  REACHABLE deficit (best-BW roof)            {at_rate - at_best:7.3f} ms  "
      f"= {100 * (at_rate - at_best) / (at_rate - at_campaign):.0f} % of it")
print(f"  REACHABLE deficit (same-mix roof)           {at_rate - at_mix:7.3f} ms  "
      f"= {100 * (at_rate - at_mix) / (at_rate - at_campaign):.0f} % of it")
print()

print("BFP8_B: identical FLOPs, 0.53x the bytes. If a site is traffic-bound this MUST show up.")
for base, b8 in (("A_inproj_DRAM", "A_inproj_bfp8"), ("B_contract_DRAM", "B_contract_bfp8"),
                 ("C_outproj_linear", "C_outproj_bfp8")):
    if b8 in A:
        x, y = A[base], A[b8]
        print(f"  {base:18s} {x['serial_ms']:.4f} -> {y['serial_ms']:.4f} ms  "
              f"{x['serial_ms'] / y['serial_ms']:.3f}x | bytes {x['MB']:.1f} -> {y['MB']:.1f} MB "
              f"({x['MB'] / y['MB']:.3f}x) | {y['serial_TFLOPs']:.2f} TFLOP/s "
              f"{y['serial_GBs']:.1f} GB/s")
print()
print("GRANULE: the pair-major -> channel-major exchange, priced by route")
for k, v in A.items():
    if k.startswith("G_"):
        print(f"  {k:24s} {v['serial_ms']:8.4f} ms  {v['serial_GBs']:7.1f} GB/s  "
              f"{100 * v['serial_GBs'] / bw_best:5.1f} % of best DRAM  | {v['note']}")
print()
print("A/A twins (the session's own floor):")
for a_, b_ in (("roof_cube4096", "roof_cube4096_AA"), ("roof_dram_2R1W", "roof_dram_2R1W_AA"),
               ("A_inproj_DRAM", "A_inproj_DRAM_AA"), ("B_contract_DRAM", "B_contract_DRAM_AA"),
               ("C_outproj_linear", "C_outproj_linear_AA"),
               ("G_channel_move_0312", "G_channel_move_0312_AA")):
    if a_ in A and b_ in A:
        x, y = A[a_]["serial_ms"], A[b_]["serial_ms"]
        print(f"  {a_:22s} {100 * abs(x - y) / min(x, y):5.2f} %")
print()
print("CLOCK:", json.dumps(d["clock"]))
