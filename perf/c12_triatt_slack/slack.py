#!/usr/bin/env python3
"""The 0.7343 s of `generic_op` slack no row owns, attributed. No device, no new measurement.

Scope handed to this row: triatt_sdpa, triatt_in, triatt_out, reblock_back -- 1.8126 s in situ
against a 1.0782 s floor. `trimul_in` and `reblock_gated` belong to `c12-reblock-delete` and are
carried here only so the arithmetic can be checked; nothing is claimed on them.

Inputs, all committed:
  perf/c12_genop_rate/insitu_sites.json         in-situ per-site ms/call and per-program legs
  perf/c12_genop_rate/headroom.json             the roofs and the floors as the brief states them
  wk/c12-generic-op-attribution genop_audit.json  per-site charged bytes AND repair_B
  perf/ttx_reblock_bfp8/byte_sensitivity_qb2c3.json  the fp32-vs-bf16 width ablation, same part

Three defects in the inputs, each checked against first principles rather than argued:

1. Five of six sites' charged bytes reconcile to a 2R+1W or 1R+1W count on the executed shape to
   within 0.25 %. `triatt_out` is charged exactly 2.00x less. The audit already says why and prints
   the number: `repair_B` = 0.0188 TB, "DROPPED by the published rule (L1 pre-allocated
   destination)". 280 of the 560 out-proj calls land their output in L1, `moves_dram` is false for
   them, and the rule charges them nothing -- not even their DRAM read.
   `insitu_sites.py:47` takes `B` and drops `repair_B`, so the one site the brief calls the worst
   offender is the one site whose byte count is halved.

2. The L1 write is not free, so it is moved bytes. The two out-proj programs in a layer are a
   natural A/B for exactly this: one writes DRAM, one writes L1, and they differ by 67.1 MB of DRAM
   traffic. A DRAM-bytes floor predicts a 0.1540 ms gap. The legs measure 0.0161 ms, 10.5 % of it,
   in both the Pairformer and the MSA leg independently. So the write costs the same wherever it
   lands, and the floor that fits is moved bytes, not DRAM bytes.

3. `reblock_gated` is charged 3 tensor-units a call (134.2 MB read, 67.1 MB written) -- a 2:1
   read:write mix -- but `headroom.py:33` roofs it against `bw_clone`, the 1R+1W arm. At its own
   matching roof it is at 71.7 % of roof, not 79.39 %, which retires the campaign's
   existence-proof efficiency. Reported, not used: that site is c12-reblock-delete's.

The third floor term: packer tile passes. `sdpa_standard` makes three full-size PACK passes over
the score matrix per k chunk (compute_common.hpp:1899 the QK^T write, :1990 the mask add, :2016 the
sub-exp), and the score matrix is 16x the op's output. That term scales with TILE COUNT, so neither
the byte nor the FLOP roof carries it. The only measured packer rate this campaign owns is 71.3
ns/tile on Wormhole (`packer-tile-passes-third-roof-neither-byte-nor-flop`, dose-responsive, linear
to 4 %). NO Blackhole packer rate has been measured, so it is carried as a band: 71.3 ns scaled by
the clock ratio at the bottom, 71.3 ns unscaled at the top. That band is this row's one transferred
number and the single thing a device arm would close.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PERF = HERE.parent
MHZ = 1350.0
T = 67108864.0          # one [1,512,512,128] bf16 pair tensor, the unit every charge lands on
WT = 32768.0            # a [128,128] bf16 weight

ins = {r["site"]: r for r in json.load(open(PERF / "c12_genop_rate" / "insitu_sites.json"))["table"]}
legs = json.load(open(PERF / "c12_genop_rate" / "insitu_sites.json"))["legs"]
head = {r["site"]: r for r in json.load(open(PERF / "c12_genop_rate" / "headroom.json"))["sites"]}
roofs = json.load(open(PERF / "c12_genop_rate" / "headroom.json"))["roofs"]
BW_2R1W, BW_1R1W = roofs["bw_2r1w"], roofs["bw_1r1w"]
FPU = roofs["fpu"]
AUD = json.loads((HERE / "genop_audit_4ed84475d.json").read_text())
aud = {s["site"]: s for s in AUD["sites"]}
AUD_SITE = {"trimul_in": "in-proj", "reblock_gated": "reblock-gated", "reblock_back": "reblock-back",
            "triatt_in": "in-proj", "triatt_sdpa": "fused SDPA", "triatt_out": "out-proj"}
audit = {}
for s in AUD["sites"]:
    for k, v in AUD_SITE.items():
        if s["site"] == v and ((k.startswith("tri") and k.startswith("triatt")) ==
                               (s["owner"] == "TriangleAttention")):
            audit[k] = s

SCOPE = ["triatt_sdpa", "triatt_in", "triatt_out", "reblock_back"]
OTHER = ["trimul_in", "reblock_gated"]

# --- 1. the charged bytes against a first-principles count on the executed shape ---------------
# rows are [b, M, K, N, FLOP] from the executed graph (TF.op_shape_rows), carried in the audit
FIRST = {   # site -> (label, first-principles DRAM+moved bytes per call, read:write mix)
    "trimul_in":     ("x[1T] + w[128x640] -> xw+gate[5T]",            1 * T + 640 * 64 * 2 + 5 * T, "2R+1W"),
    "triatt_in":     ("x[1T] + w[128x544] -> q|k|v|g[4T] + bias[T/32]", 1 * T + 544 * 64 * 2 + 4.25 * T, "2R+1W"),
    "triatt_sdpa":   ("q,k,v[3T] + bias[T/32] -> out[1T]",            3 * T + T / 32 + 1 * T, "3R+1W"),
    "triatt_out":    ("gated[1T] + w[128x128] -> out[1T]",            1 * T + WT + 1 * T, "2R+1W"),
    "reblock_gated": ("xw[2T] -> gated[1T]",                          2 * T + 1 * T, "2R+1W"),
    "reblock_back":  ("x[1T] -> y[1T]",                               1 * T + 1 * T, "1R+1W"),
}
print("=== SITES: charged bytes vs a first-principles count on the executed shape ===")
print("%-14s %6s %10s %10s %10s %7s %-6s %s"
      % ("site", "calls", "chargedMB", "repairMB", "firstMB", "chg/1st", "mix", "what"))
for s in OTHER + SCOPE:
    a, n = audit[s], ins[s]["calls"]
    lab, fp, mix = FIRST[s]
    print("%-14s %6d %10.3f %10.3f %10.3f %7.4f %-6s %s"
          % (s, n, a["B"] / n / 1e6, a["repair_B"] / n / 1e6, fp / 1e6,
             (a["B"] / n) / fp, mix, lab))

# --- 2. the natural A/B inside triatt_out: one program writes DRAM, one writes L1 --------------
print("\n=== the out-proj A/B: 280 DRAM destinations and 280 L1 destinations ===")
n = ins["triatt_out"]["calls"]
print("charged W %.4f T-units/call = %.1f full writes over %d calls -> %d DRAM destinations"
      % (audit["triatt_out"]["W"] / n / T, audit["triatt_out"]["W"] / T, n,
         round(audit["triatt_out"]["W"] / T)))
print("repair_B  %.4f T-units/call = %.1f dropped reads          -> %d L1 destinations"
      % (audit["triatt_out"]["repair_B"] / n / T, audit["triatt_out"]["repair_B"] / T,
         round(audit["triatt_out"]["repair_B"] / T)))
print("one per layer (264 Pairformer + 16 MSA = 280), i.e. one of the two directions.")
ab = {}
for leg, rows in legs.items():
    d = {r["idx"]: r["ms"] for r in rows if r["site"] == "triatt_out"}
    ab[leg] = d
    print("  %-11s idx10 %.6f ms   idx13 %.6f ms   delta %+.6f ms  (%+.2f %%)"
          % (leg, d[10], d[13], d[13] - d[10], 100 * (d[13] / d[10] - 1)))
d10 = sum(v[10] for v in ab.values()) / len(ab)
d13 = sum(v[13] for v in ab.values()) / len(ab)
gap_meas = d13 - d10
gap_pred = T / (BW_2R1W * 1e9) * 1e3
print("  mean       idx10 %.6f   idx13 %.6f   measured gap %.6f ms" % (d10, d13, gap_meas))
print("  a DRAM-BYTES floor predicts %.6f ms for 67.1 MB at %.2f GB/s." % (gap_pred, BW_2R1W))
print("  measured / predicted = %.2f %%.  The write costs the same wherever it lands, so the"
      % (100 * gap_meas / gap_pred))
print("  floor that fits this site is MOVED bytes, not DRAM bytes. Both legs agree independently.")

# --- 3. the packer term, the third floor ------------------------------------------------------
print("\n=== the third floor term: packer tile passes over the SDPA score matrix ===")
B, SQT, SKT, KCH, CORES, PACKS = 2048, 16, 16, 1, 110, 3
score_tiles = B * SQT * SKT * KCH
per_core = score_tiles / CORES
NS_WH = 71.3                       # MEASURED on Wormhole, dose-responsive, linear to 4 %
NS_LO = NS_WH * 1000.0 / MHZ       # the same packer scaled by the clock ratio
print("score matrix %d x %dx%d tiles x %d k chunk = %d tiles; %d cores -> %.1f tiles/core"
      % (B, SQT, SKT, KCH, score_tiles, CORES, per_core))
print("%d full-size PACK passes/chunk: compute_common.hpp:1899 QK^T write, :1990 mask add, "
      ":2016 sub-exp" % PACKS)
pk = {}
for tag, ns in (("clock-scaled", NS_LO), ("Wormhole-measured", NS_WH)):
    ms = per_core * PACKS * ns / 1e6
    pk[tag] = ms
    print("  at %5.1f ns/tile (%-17s): %.4f ms/call = %.1f %% of the op's %.4f ms"
          % (ns, tag, ms, 100 * ms / ins["triatt_sdpa"]["ms"], ins["triatt_sdpa"]["ms"]))
print("  traffic floor %.4f ms (%.1f %%), arithmetic floor %.4f ms (%.1f %%)."
      % (head["triatt_sdpa"]["traf_ms"],
         100 * head["triatt_sdpa"]["traf_ms"] / ins["triatt_sdpa"]["ms"],
         head["triatt_sdpa"]["arith_ms"],
         100 * head["triatt_sdpa"]["arith_ms"] / ins["triatt_sdpa"]["ms"]))
print("  The packer term is the LARGEST of the three at both ends of the band and is the one term")
print("  the campaign's max(traffic, arithmetic) floor does not carry.")

# --- 4. reblock_back: the width ablation splits byte cost from transaction cost ----------------
print("\n=== reblock_back: a same-part width ablation splits the cost ===")
bs = json.load(open(PERF / "ttx_reblock_bfp8" / "byte_sensitivity_qb2c3.json"))
row = [r for r in bs["rows"] if r["direction"] == "back" and r["n"] == 512 and r["c"] == 128][0]
t2, t4 = row["bf16_ctrl_ms"], row["fp32_ms"]
b_per = (t4 - t2) / 2.0
a_fix = t2 - 2 * b_per
print("host %s dev %s grid %s  back/512/128 DRAM->DRAM: bf16 %.5f ms, fp32 %.5f ms, ratio %.4f, "
      "spread %.4f" % (bs["host"], bs["visible_devices"], bs["grid"], t2, t4,
                       row["fp32_over_bf16"], row["bf16_spread"]))
print("t(w) = a + b*w  ->  byte term %.5f ms (%.1f %%), width-INDEPENDENT term %.5f ms (%.1f %%)"
      % (2 * b_per, 100 * 2 * b_per / t2, a_fix, 100 * a_fix / t2))
ms_fold = ins["reblock_back"]["ms"]
fl = head["reblock_back"]["floor_ms"]
lo_frac, hi_frac = a_fix / t2, (ms_fold - fl) / ms_fold
print("standalone %.5f ms runs %.3fx the in-fold %.5f ms, so the TIME does not transfer -- the "
      "FRACTION is" % (t2, t2 / ms_fold, ms_fold))
print("in fold: DRAM floor %.5f ms, so the non-byte residual is at most %.1f %% (%.5f ms)."
      % (fl, 100 * hi_frac, ms_fold - fl))
print("transaction term is %.1f-%.1f %% of the site = %.5f-%.5f ms/call = %.4f-%.4f s of fold,"
      % (100 * lo_frac, 100 * hi_frac, lo_frac * ms_fold, hi_frac * ms_fold,
         lo_frac * ms_fold * 560 / 1e3, hi_frac * ms_fold * 560 / 1e3))
print("i.e. %.0f-%.0f %% of the site's whole %.4f s slack."
      % (100 * lo_frac * ms_fold * 560 / 1e3 / (ins["reblock_back"]["sec"] - head["reblock_back"]["floor_sec"]),
         100 * hi_frac * ms_fold * 560 / 1e3 / (ins["reblock_back"]["sec"] - head["reblock_back"]["floor_sec"]),
         ins["reblock_back"]["sec"] - head["reblock_back"]["floor_sec"]))
ntx = 32768 / CORES * 64
print("the gather is 64 transactions per output tile whatever the kernel structure "
      "(reader_reblock_permute_back.cpp:30) -> %.0f 32-byte L1->L1 reads/core/call"
      % ntx)
print("that prices the transaction at %.2f-%.2f ns = %.1f-%.1f cycles at %.0f MHz."
      % (lo_frac * ms_fold * 1e6 / ntx, hi_frac * ms_fold * 1e6 / ntx,
         lo_frac * ms_fold * 1e-3 * MHZ * 1e6 / ntx, hi_frac * ms_fold * 1e-3 * MHZ * 1e6 / ntx, MHZ))

# --- 5. the corrected table -------------------------------------------------------------------
print("\n=== DEFICIT: the four sites, corrected ===")
CORR = {}
for s in OTHER + SCOPE:
    h, i = head[s], ins[s]
    mb = FIRST[s][1] / 1e6
    bw = BW_1R1W if FIRST[s][2] == "1R+1W" else BW_2R1W
    traf = mb * 1e6 / (bw * 1e9) * 1e3
    arith = h["arith_ms"]
    CORR[s] = dict(traf=traf, arith=arith, floor=max(traf, arith), ms=i["ms"], calls=i["calls"],
                   sec=i["sec"], bw=bw, mb=mb)
print("%-14s %6s %8s %9s %8s %8s %8s %7s %7s  %s"
      % ("site", "calls", "ms/call", "roof GB/s", "MB/call", "trafms", "floorms", "%roof",
         "GB/s", "was %roof"))
for s in OTHER + SCOPE:
    c = CORR[s]
    print("%-14s %6d %8.4f %9.2f %8.3f %8.4f %8.4f %6.2f%% %7.1f  %6.2f%%"
          % (s, c["calls"], c["ms"], c["bw"], c["mb"], c["traf"], c["floor"],
             100 * c["floor"] / c["ms"], c["mb"] * 1e6 / (c["ms"] * 1e-3) / 1e9,
             head[s]["pct_roof"]))
tot_s = sum(CORR[s]["sec"] for s in SCOPE)
tot_f = sum(CORR[s]["floor"] * CORR[s]["calls"] / 1e3 for s in SCOPE)
brief_f = sum(head[s]["floor_sec"] for s in SCOPE)
print("\nthe four sites: in situ %.4f s / %.1f Mc" % (tot_s, tot_s * MHZ))
print("floor as the brief states it   %.4f s -> slack %.4f s / %.1f Mc"
      % (brief_f, tot_s - brief_f, (tot_s - brief_f) * MHZ))
print("floor with the byte model fixed %.4f s -> slack %.4f s / %.1f Mc   (%.4f s was artifact)"
      % (tot_f, tot_s - tot_f, (tot_s - tot_f) * MHZ, brief_f - tot_f))
for tag, ms in pk.items():
    f3 = tot_f - CORR["triatt_sdpa"]["floor"] * 560 / 1e3 + max(ms, CORR["triatt_sdpa"]["floor"]) * 560 / 1e3
    print("  + the packer term at the %-17s end: floor %.4f s -> slack %.4f s / %.1f Mc"
          % (tag, f3, tot_s - f3, (tot_s - f3) * MHZ))

# --- 5b. what minimal_matmul's 1.32x actually is ------------------------------------------------
print("\n=== MECHANISM at the two minimal_matmul sites: one rate, and no roof to score it against ===")
for s in ("triatt_in", "triatt_out"):
    c = CORR[s]
    print("  %-11s %8.3f MB/call in %.4f ms = %.1f GB/s = %.2f %% of bw_add8192"
          % (s, c["mb"], c["ms"], c["mb"] * 1e6 / (c["ms"] * 1e-3) / 1e9,
             100 * c["floor"] / c["ms"]))
print("  2.62x apart in bytes, 1 output buffer vs 5, and 0.8 % apart in rate. Both a per-call")
print("  fixed cost and a multi-destination page cost are refuted by that pair.")
print("  bw_add8192 is ttnn.add on a big DRAM tensor (roofs.py:119): unicast, page-sized, no")
print("  relay. minimal_matmul's in0 is a DAISY-CHAIN RELAY -- dm_in0_sender.cpp:275-296 is a")
print("  per-block semaphore wait, a whole-block unicast to the next core, a Blackhole-only")
print("  noc_async_writes_flushed() and a remote semaphore set, once per k block per hop.")
print("  Every cube arm in this session is ARITHMETIC-bound (cube4096 113.68 TFLOP/s at 83.3")
print("  GB/s, cube2048_dflt 101.38 TFLOP/s at 148.5 GB/s), so the session measured NO streaming")
print("  roof for a block-relayed matmul dataflow. 1.32x against an eltwise roof is not a")
print("  measured deficit: it is a missing roof. Same defect class as the dense-cube roof that")
print("  self-refuted c12-matmul-key-attribution's 78.24 TFLOP/s ceiling, byte side instead of")
print("  FLOP side.")

# --- 6. the existence proof, re-derived -------------------------------------------------------
print("\n=== the existence-proof efficiency, re-derived ===")
eff = {s: 100 * CORR[s]["floor"] / CORR[s]["ms"] for s in OTHER + SCOPE}
best = max(eff, key=eff.get)
print("corrected per-site % of its OWN matching roof: "
      + ", ".join("%s %.2f%%" % (s, eff[s]) for s in sorted(eff, key=lambda x: -eff[x])))
print("best is %s at %.2f %%, with triatt_in at %.2f %% -- the SAME kernel (minimal_matmul) at two"
      % (best, eff[best], eff["triatt_in"]))
print("shapes %.2fx apart in bytes, streaming %.1f and %.1f GB/s. That is a rate, not a per-call"
      % (CORR["triatt_in"]["mb"] / CORR["triatt_out"]["mb"],
         CORR["triatt_out"]["mb"] * 1e6 / (CORR["triatt_out"]["ms"] * 1e-3) / 1e9,
         CORR["triatt_in"]["mb"] * 1e6 / (CORR["triatt_in"]["ms"] * 1e-3) / 1e9))
print("cost, and it refutes the brief's leading hypothesis for triatt_out.")
print("the brief's 79.39 % (reblock_gated) is retired: that site is 2R+1W and was roofed 1R+1W.")
for f, why in ((eff[best] / 100, "the corrected best any of the six reaches"),
               (0.7939, "the brief's stated ceiling, now refuted")):
    print("  prize at f = %.4f [%s]: %.4f s -> %.4f s, %.4f s / %.1f Mc"
          % (f, why, tot_s, tot_f / f, tot_s - tot_f / f, (tot_s - tot_f / f) * MHZ))
f3 = tot_f - CORR["triatt_sdpa"]["floor"] * 560 / 1e3 + pk["Wormhole-measured"] * 560 / 1e3
print("  with the packer term at the Wormhole-measured end, f = %.4f needs %.4f s against %.4f s"
      % (eff[best] / 100, f3 / (eff[best] / 100), tot_s))
print("  in situ: the four sites are ALREADY at or past that efficiency, prize %.4f s."
      % (tot_s - f3 / (eff[best] / 100)))

json.dump(dict(scope=SCOPE, corrected={s: CORR[s] for s in OTHER + SCOPE}, eff=eff,
               insitu_s=tot_s, floor_brief_s=brief_f, floor_fixed_s=tot_f,
               packer_ms=pk, out_proj_ab=dict(idx10=d10, idx13=d13, gap_meas=gap_meas,
                                              gap_pred_dram=gap_pred),
               reblock_back_tx_frac=[lo_frac, hi_frac], mhz=MHZ),
          open(HERE / "slack.json", "w"), indent=1)
