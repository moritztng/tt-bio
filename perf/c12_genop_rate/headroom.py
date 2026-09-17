#!/usr/bin/env python3
"""Per-site headroom for generic_op, against roofs measured on the same part in the same session.

Joins `insitu_sites.json` (in-fold per-site seconds out of c12-profiled-fold's ops reports) to
`runs/roof1/roofs_qb2c3_r40.json` (this row's own roof session on qb2 card 3) and prices, per site:

  traffic floor     bytes / the roof whose READ:WRITE MIX matches the site. A matmul is 2R + 1W so
                    it gets bw_add8192; a reblock is 1R + 1W so it gets bw_clone. The two differ by
                    1.11x on this part, which is why one number cannot serve both.
  arithmetic floor  FLOP / the cube roof AT THE FIDELITY THE SITE RUNS AT, read out of the ops
                    report's MATH FIDELITY column, not assumed.
  floor             max(traffic, arithmetic), the same construction the census uses.

Prize is evaluated at three efficiencies: 1.00 (every site at its own roof, which nothing on this
part reaches), 0.90, and the best efficiency any of the six sites actually achieves today, which is
the only one with an empirical existence proof.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MHZ = 1350.0
ins = json.load(open(HERE / "insitu_sites.json"))
roof = json.load(open(HERE / "runs" / "roofs_qb2c3_r40.json"))
R = {r["arm"]: r for r in roof["rows"]}

BW_2R1W = R["bw_add8192"]["GBs"]          # matmul-shaped: two reads and a write
BW_1R1W = R["bw_clone"]["GBs"]            # reblock-shaped: one read and a write
FPU = {"HiFi4": R["cube4096"]["TFLOPs"],  # packer_l1_acc on, the faster arm at each fidelity
       "HiFi2": R["cube4096_r_hifi2"]["TFLOPs"],
       "LoFi": R["cube4096_lofi"]["TFLOPs"]}
MIX = {"trimul_in": BW_2R1W, "triatt_in": BW_2R1W, "triatt_sdpa": BW_2R1W,
       "triatt_out": BW_2R1W, "reblock_gated": BW_1R1W, "reblock_back": BW_1R1W}

print("roofs, qb2 card 3, 110 of 110 cores, AICLK forced and sampled at 1350 MHz")
print("  streaming 2R+1W  %.1f GB/s   (A/A %.1f, floor %.2f%%)"
      % (BW_2R1W, R["bw_add8192_aa"]["GBs"],
         100 * abs(R["bw_add8192_aa"]["GBs"] - BW_2R1W) / BW_2R1W))
print("  streaming 1R+1W  %.1f GB/s   (A/A %.1f, floor %.2f%%)"
      % (BW_1R1W, R["bw_clone_aa"]["GBs"],
         100 * abs(R["bw_clone_aa"]["GBs"] - BW_1R1W) / BW_1R1W))
print("  cube4096 HiFi4   %.2f TFLOP/s (A/A %.2f, floor %.2f%%)"
      % (FPU["HiFi4"], R["cube4096_aa"]["TFLOPs"],
         100 * abs(R["cube4096_aa"]["TFLOPs"] - FPU["HiFi4"]) / FPU["HiFi4"]))
print("  cube4096 HiFi2   %.2f TFLOP/s   cube4096 LoFi %.2f TFLOP/s"
      % (FPU["HiFi2"], FPU["LoFi"]))
print("  machine balance at HiFi4: %.1f FLOP/byte on the 2R+1W roof"
      % (FPU["HiFi4"] * 1e12 / (BW_2R1W * 1e9)))
print()

hdr = ("site             calls  ms/call  MB/call GFLOP/call  F/B  fid   trafms  arithms  "
       "floorms   ratio  %roof")
print(hdr)
tot = dict(sec=0.0, floor=0.0)
out = []
for t in ins["table"]:
    site, ms, mb, gf = t["site"], t["ms"], t["mb_call"], t["gflop_call"]
    fid = t["fid"][0]
    traf_ms = mb * 1e6 / (MIX[site] * 1e9) * 1e3
    arith_ms = gf * 1e9 / (FPU[fid] * 1e12) * 1e3 if gf else 0.0
    floor_ms = max(traf_ms, arith_ms)
    sec = t["sec"]
    fl_sec = floor_ms * t["calls"] / 1e3
    tot["sec"] += sec
    tot["floor"] += fl_sec
    fpb = gf * 1e9 / (mb * 1e6) if mb else 0.0
    out.append(dict(site=site, calls=t["calls"], ms=ms, traf_ms=traf_ms, arith_ms=arith_ms,
                    floor_ms=floor_ms, sec=sec, floor_sec=fl_sec, fid=fid, fpb=fpb,
                    ratio=ms / floor_ms, pct_roof=100 * floor_ms / ms,
                    bound="traffic" if traf_ms >= arith_ms else "ARITHMETIC"))
    print("%-16s %5d %8.4f %8.1f %10.3f %5.1f %-5s %8.4f %8.4f %8.4f %7.3f %6.1f%%"
          % (site, t["calls"], ms, mb, gf, fpb, fid, traf_ms, arith_ms, floor_ms,
             ms / floor_ms, 100 * floor_ms / ms))
print("%-16s %5d %8s %8s %10s %5s %-5s %8s %8s %8.4f s %6.3f"
      % ("TOTAL", sum(o["calls"] for o in out), "", "", "", "", "", "", "",
         tot["floor"], tot["sec"] / tot["floor"]))
print()
print("in situ  %.4f s / %.1f Mc" % (tot["sec"], tot["sec"] * MHZ))
print("floor    %.4f s / %.1f Mc   (per-site max(traffic, arithmetic) at the measured roofs)"
      % (tot["floor"], tot["floor"] * MHZ))
arith_bound = [o["site"] for o in out if o["bound"] == "ARITHMETIC"]
print("sites where arithmetic binds: %s" % (arith_bound or "NONE -- all six are bandwidth-bound"))
best_eff = max(o["pct_roof"] for o in out) / 100.0
print()
for f, why in ((1.00, "every site at its own measured roof, nothing on this part reaches it"),
               (0.90, "90 % of roof"),
               (best_eff, "the best efficiency any of the six reaches today (%s)"
                % max(out, key=lambda o: o["pct_roof"])["site"])):
    tgt = tot["floor"] / f
    prize = tot["sec"] - tgt
    print("prize at f = %.3f: %.4f s -> %.4f s, prize %.4f s / %.1f Mc   [%s]"
          % (f, tot["sec"], tgt, prize, prize * MHZ, why))
json.dump(dict(roofs=dict(bw_2r1w=BW_2R1W, bw_1r1w=BW_1R1W, fpu=FPU), sites=out,
               total_s=tot["sec"], floor_s=tot["floor"], best_eff=best_eff),
          open(HERE / "headroom.json", "w"), indent=1)
