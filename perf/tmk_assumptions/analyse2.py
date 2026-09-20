#!/usr/bin/env python3
"""Price the shipped channel move from granule2.json + census_gated.json.

Everything here is arithmetic over two measured files. Nothing is retyped by hand.
"""
import json
from pathlib import Path

H = Path(__file__).parent
G = json.loads((H / "granule2.json").read_text())
C = json.loads((H / "census_gated.json").read_text())
A = G["arms"]
N, D = G["shape"]["N"], G["shape"]["D"]
Z = N * N * D * 2                                   # one channel-slice plane, bytes
ms = lambda k: A[k]["serial_ms"]
gbs = lambda k: A[k]["serial_GBs"]

out = []
p = out.append
p(f"shape N={N} D={D}  Z={Z/1e6:.3f} MB   clock {G['clock']['min']}-{G['clock']['max']} MHz, "
  f"n={G['clock']['n']} samples, {G['clock']['errors']} errors, span {G['clock']['sampler_span_s']}"
  f"/{G['clock']['measure_span_s']} s")
p("")
p(f"{'arm':26s} {'ms':>8s} {'GB/s':>8s} {'MB':>9s}  A/A")
for k in A:
    if k.endswith("_AA"):
        continue
    aa = A.get(k + "_AA")
    d = f"{abs(ms(k+'_AA')-ms(k))/ms(k)*100:.2f} %" if aa else "-"
    p(f"{k:26s} {ms(k):8.4f} {gbs(k):8.1f} {A[k]['MB']:9.3f}  {d}")
p("")

copy = gbs("C_clone") * 1e9                          # B/s, the same bytes with no exchange
tile = gbs("C_tile_transpose") * 1e9
for k, byts in (("P_gated_production", 3 * Z), ("M_reblock_back", 2 * Z)):
    floor_ms = byts / copy * 1e3
    p(f"{k}: {ms(k):.4f} ms vs a pure-copy floor of {floor_ms:.4f} ms on the same "
      f"{byts/1e6:.1f} MB -> {ms(k)/floor_ms:.3f}x above copy, {ms(k)-floor_ms:.4f} ms recoverable")
    p(f"{' '*len(k)}  against the tile-granular transpose rate: "
      f"{byts/tile*1e3:.4f} ms -> {ms(k)/(byts/tile*1e3):.3f}x")

fwd_floor = 3 * Z / copy * 1e3
back_floor = 2 * Z / copy * 1e3
n_gated = C["totals"]["gated"]
n_back = C["totals"]["back"]
fold = C["fold_s"] * 1e3
mod_meas = 2 * ms("P_gated_production") + ms("M_reblock_back")
mod_floor = 2 * fwd_floor + back_floor
p("")
p(f"per trimul call ({n_gated//n_back} gated moves + 1 back move, from the census):")
p(f"  measured {mod_meas:.3f} ms   pure-copy floor {mod_floor:.3f} ms   "
  f"recoverable {mod_meas-mod_floor:.3f} ms ({(mod_meas-mod_floor)/mod_meas*100:.1f} % of the term)")
p(f"  trix-radical booked 2.551 ms for this term in an 11.519 ms/call module: "
  f"measured share {mod_meas/11.519*100:.1f} %, recoverable share {(mod_meas-mod_floor)/11.519*100:.1f} %")
p("")
p(f"per {C['fixture']} fold ({fold:.0f} ms, {n_gated} gated + {n_back} back calls, derived as "
  f"count x standalone per-call time):")
tot = n_gated * ms("P_gated_production") + n_back * ms("M_reblock_back")
rec = n_gated * (ms("P_gated_production") - fwd_floor) + n_back * (ms("M_reblock_back") - back_floor)
p(f"  channel moves {tot:.0f} ms = {tot/fold*100:.2f} % of the fold")
p(f"  recoverable to a pure copy {rec:.0f} ms = {rec/fold*100:.2f} % -> fold {fold/(fold-rec):.4f}x "
  f"as an unreachable upper bound")
p("")
p("the three numbers in circulation for 'the shipped move', all at N=512 D=128 in this session:")
p(f"  ttnn.permute(0,3,1,2)      {gbs('M_permute_stock_0312'):6.1f} GB/s  (pass 1 read 63.3 -- reproduced, "
  f"but production calls it 0 times per fold)")
p(f"  reblock_permute (ours)     {gbs('M_reblock_fwd'):6.1f} GB/s  (trix-radical's 158; perfwar's 221 was N=1024)")
p(f"  reblock_permute_gated      {gbs('P_gated_production'):6.1f} GB/s  (THE shipped call, {n_gated} of {n_gated} per fold)")
p(f"  tile-granular control      {gbs('C_tile_transpose'):6.1f} GB/s")
p(f"  straight clone             {gbs('C_clone'):6.1f} GB/s")
p("")
p(f"what the fusion already banked: the sequence it replaces reads "
  f"{ms('P_unfused_stock_permute'):.3f} ms with the stock move and "
  f"{ms('P_unfused_reblock'):.3f} ms with ours, against {ms('P_gated_production'):.3f} ms fused: "
  f"{ms('P_unfused_stock_permute')/ms('P_gated_production'):.2f}x / "
  f"{ms('P_unfused_reblock')/ms('P_gated_production'):.2f}x")
txt = "\n".join(out)
(H / "analyse2.txt").write_text(txt + "\n")
print(txt)
