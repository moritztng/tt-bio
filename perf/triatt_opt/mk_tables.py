#!/usr/bin/env python3
"""Build the triatt-absolute-optimal tables from the two measured JSONs. No hand arithmetic."""
import json, sys
from pathlib import Path

R = Path("/home/ttuser/.coworker/wt/triatt-absolute-optimal")
ops = json.load(open(R / "perf/triatt_opt/ops_pv2_512_qb2c2.json"))
fl = json.load(open(R / "perf/triatt_opt/floor_qb2c2.json"))
MiB = 2 ** 20
N, C, H, D = 512, 256, 8, 32
Z = N * N * C * 2 / MiB          # one pair tensor, MiB

# ---- roofs on this card, at the pair-tensor size -------------------------------------------
roof = {}
for r in fl["roofs"]:
    if r.get("mb") == 128 and "gbps" in r:
        roof[r["roof"]] = r["gbps"]
for r in fl["roofs"]:
    if r.get("mb") == 52 and r["roof"] == "l1_copy" and "gbps" in r:
        roof["l1_copy@52MB"] = r["gbps"]
print("ROOFS (card 2, 128 MB clone unless noted):")
for k, v in roof.items():
    print(f"  {k:16s} {v:7.1f} GB/s")

# ---- the two spans -------------------------------------------------------------------------
# traffic model per op, DERIVED from shapes: (read MiB, write MiB)
TRAF = {
    "layer_norm@2189":      (Z, Z),
    "linear@2197":          (Z, N * N * 32 * 2 / MiB),
    "permute@2201":         (N * N * 32 * 2 / MiB, N * N * H * 2 / MiB),
    "minimal_matmul@2313":  (Z, 3 * Z),
    "minimal_matmul@2320":  (Z, Z),
    "nlp_create_qkv_heads@2209": (3 * Z, 3 * Z),
    "scaled_dot_product_attention@433": (3 * Z + N * (N * N * H * 2 / MiB), Z),
    "nlp_concat_heads@2237": (Z, Z),
    "multiply_@2243":       (2 * Z, Z),
    "linear@2245":          (Z, Z),
    "to_layout@678":        (Z, Z),
    "permute@679":          (Z, Z),
    "to_layout@681":        (Z, Z),
}
spans = {"start": "tenstorrent.py:2868", "end": "tenstorrent.py:2872"}
tot = {}
for name, tag in spans.items():
    rows = [r for r in ops["records"] if r["chain"] and r["chain"][-1] == tag]
    s = sum(r["s"] for r in rows)
    tot[name] = s
    print(f"\n=== triatt_{name}  MEASURED {s*1e3:.3f} ms  ({len(rows)} ops) ===")
    print(f"{'op@site':34s} {'ms':>8s} {'read':>8s} {'write':>8s} {'GB/s':>8s} {'%roof':>7s}")
    tr_tot = 0.0
    for r in rows:
        key = f"{r['op']}@{r['site'].split(':')[1]}"
        t = TRAF.get(key)
        if t:
            mb = (t[0] + t[1]) * MiB / 1e6
            gbps = mb / (r["s"] * 1e3) if r["s"] else 0
            pct = 100 * gbps / roof["dram_copy"]
            tr_tot += t[0] + t[1]
            print(f"{key:34s} {r['s']*1e3:8.3f} {t[0]:8.1f} {t[1]:8.1f} {gbps:8.1f} {pct:6.1f}%")
        else:
            print(f"{key:34s} {r['s']*1e3:8.3f} {'-':>8s} {'-':>8s} {'view':>8s} {'':>7s}")
    print(f"{'TRAFFIC TOTAL (MiB)':34s} {'':8s} {tr_tot:17.1f}")
    tot[name + "_traffic"] = tr_tot

blk = tot["start"] + tot["end"]
print(f"\ntriatt per Pairformer block MEASURED = {blk*1e3:.3f} ms")
print(f"block wall MEASURED                  = {ops['block_wall_s']*1e3:.3f} ms")
print(f"triatt share of block                = {100*blk/ops['block_wall_s']:.1f} %")
print(f"traffic per block DERIVED            = {tot['start_traffic']+tot['end_traffic']:.0f} MiB")
PF_STACK = 58.137          # s, MEASURED, moonshot-4x-512aa-ledger
FOLD = 79.172              # s, MEASURED, same source
BLOCKS = 528
print(f"pf.triatt per fold = share x pf_stack = {blk/ops['block_wall_s']*PF_STACK:.3f} s"
      f"  ({100*blk/ops['block_wall_s']*PF_STACK/FOLD:.1f} % of the {FOLD} s fold)")
print(f"pf.triatt per fold = {BLOCKS} x block  = {BLOCKS*blk:.3f} s")
print(f"the brief's DERIVED figure            = 12.871 s "
      f"(ratio {BLOCKS*blk/12.871:.3f}x)")

# ---- the floor ------------------------------------------------------------------------------
print("\n=== DERIVED byte floor, one triatt call, N=512 c=256 h=8 d=32 bf16 ===")
bias = N * N * H * 2 / MiB
floor = {"z read (pass 1, for the bias)": Z, "bias write": bias,
         "z read (pass 2, rows)": Z, "bias read, multicast per head group": bias,
         "output write": Z}
for k, v in floor.items():
    print(f"  {k:44s} {v:8.1f} MiB")
fb = sum(floor.values())
fb_percore = fb - bias + 110 * (N * N * 2 / MiB)      # 110 cores x one head slice
print(f"  {'TOTAL (bias multicast)':44s} {fb:8.1f} MiB")
print(f"  {'TOTAL (bias once per core, 110 cores)':44s} {fb_percore:8.1f} MiB")
rd = 2 * Z + bias
wr = Z + bias
print(f"\nmemory time floor: read {rd:.0f} MiB / {roof['dram_read']:.1f} = "
      f"{rd*MiB/1e6/roof['dram_read']:.3f} ms; write {wr:.0f} MiB / {roof['dram_write']:.1f} = "
      f"{wr*MiB/1e6/roof['dram_write']:.3f} ms; combined stream "
      f"{fb*MiB/1e6/roof['dram_copy']:.3f} ms")

# compute floor from MEASURED component times
mm = {r.get("combo"): r for r in fl["fused"] if "combo" in r}
sd = {(r.get("B"), r.get("qkv_mem"), r.get("bias", "on")): r for r in fl["sdpa"]}
proj = mm["all_1056"]["ms"]
att = sd[(512, "DRAM", "off")]["ms"]
out = [r for r in fl["mm"] if r["M"] == 262144 and r["N"] == 256][0]["ms"]
print(f"\ncompute floor from MEASURED op times:")
print(f"  fused projection qkv+gate+bias N=1056   {proj:.3f} ms  MEASURED")
print(f"  attention, bias off, wide-q             {att:.3f} ms  MEASURED")
print(f"  out projection, (4,8,1,4,1)             {out:.3f} ms  MEASURED")
cf = proj + att + out
print(f"  TOTAL compute floor                     {cf:.3f} ms/call = {2*cf:.3f} ms/block")
print(f"  ceiling against the measured block      {blk*1e3/(2*cf):.3f}x")
print(f"  sanity: {fb*MiB/1e6:.1f} MB / {cf:.3f} ms = {fb*MiB/1e6/cf:.1f} GB/s "
      f"= {100*fb*MiB/1e6/cf/roof['dram_copy']:.1f} % of the copy roof")

# ---- lever prices ---------------------------------------------------------------------------
print("\n=== lever prices, all MEASURED unless marked ===")
def _key(r):
    site = r['site'].split(':')[1]
    if site == '984':
        site = r['chain'][1].split(':')[1]
    return f"{r['op']}@{site}"
cen = {_key(r): r['s'] * 1e3
       for r in ops['records'] if r['chain'] and r['chain'][-1] == spans['start']}
shipped3 = cen["minimal_matmul@2313"] + cen["minimal_matmul@2320"] + cen["linear@2197"]
print(f"L2 fused projection: shipped {shipped3:.3f} -> {proj:.3f} ms/call "
      f"(saves {shipped3-proj:.3f} ms/call, {2*(shipped3-proj):.3f} ms/block, "
      f"{100*2*(shipped3-proj)/(blk*1e3):.1f} % of the sub-block), bit-exact "
      f"{fl['fused_bitexact']}")
print(f"L3 out projection cfg: {cen['linear@2245']:.3f} -> {out:.3f} ms/call "
      f"(saves {cen['linear@2245']-out:.3f}, {2*(cen['linear@2245']-out):.3f} ms/block, "
      f"{100*2*(cen['linear@2245']-out)/(blk*1e3):.1f} %)")
sd_on = sd[(512, "DRAM", "on")]["ms"]
print(f"L4 bias-once: {sd_on:.3f} -> {att:.3f} ms/call ({sd_on/att:.3f}x on the op, "
      f"saves {2*(sd_on-att):.3f} ms/block, {100*2*(sd_on-att)/(blk*1e3):.1f} %)")
print(f"   marginal bias stream: {sd_on-att:.3f} ms for "
      f"{N*bias:.0f} MiB = {N*bias*MiB/1e6/(sd_on-att):.0f} GB/s "
      f"(pattern roof, vs the {roof['dram_read']:.0f} GB/s streaming read roof)")
hs = cen["nlp_create_qkv_heads@2209"]; nc = cen["nlp_concat_heads@2237"]
print(f"L5 head split {hs:.3f} + concat {nc:.3f} ms/call: "
      f"{2*(hs+nc):.3f} ms/block, {100*2*(hs+nc)/(blk*1e3):.1f} %")
etra = [r for r in ops["records"] if r["chain"] and r["chain"][-1] == spans["end"]
        and r["site"].split(":")[1] in ("678", "679", "681")]
tt = sum(r["s"] for r in etra) * 1e3
print(f"L6 pair transposes (ending only): {tt:.3f} ms/block, {100*tt/(blk*1e3):.1f} % of the "
      f"sub-block; own clone floor 2 x {2*Z*MiB/1e6/roof['dram_copy']:.3f} ms")
gm = cen["multiply_@2243"]
print(f"L7 gate multiply {gm:.3f} ms/call, {3*Z*MiB/1e6/(gm):.1f} GB/s = "
      f"{100*3*Z*MiB/1e6/gm/roof['dram_copy']:.1f} % of the copy roof (saturated)")
ln = cen["layer_norm@2189"]
print(f"L8 layer_norm {ln:.3f} ms/call, {2*Z*MiB/1e6/ln:.1f} GB/s = "
      f"{100*2*Z*MiB/1e6/ln/roof['dram_copy']:.1f} % of the copy roof; floor "
      f"{2*Z*MiB/1e6/roof['dram_copy']:.3f} ms")

# ---- best designed configuration -------------------------------------------------------------
best_call = ln + proj + cen["permute@2201"] + 0.10 + att + gm + out + 0.013
print(f"\nbest designed config (start variant) = {best_call:.3f} ms/call vs "
      f"{tot['start']*1e3:.3f} MEASURED = {tot['start']*1e3/best_call:.3f}x")
print(f"  block (transposes absorbed)        = {2*best_call:.3f} ms vs {blk*1e3:.3f} = "
      f"{blk*1e3/(2*best_call):.3f}x")
print(f"  residual to the floor              = {2*best_call - 2*cf:.3f} ms/block")

# ---- chunking / L1 residency verdict ----------------------------------------------------------
print("\n=== L1-resident row-chunk verdict (P2/P3) ===")
for N_ in (768, 256):
    full = [r for r in fl["mm"] if r["M"] == 262144 and r["N"] == N_][0]
    print(f" N={N_}: production unchunked DRAM->DRAM {full['ms']:.4f} ms "
          f"({full['tflops']:.1f} TF/s)")
    for R_ in (32, 64):
        for s, d in (("DRAM", "DRAM"), ("L1", "L1")):
            c = [r for r in fl["mm"] if r.get("row_chunk") == R_ and r["N"] == N_
                 and r.get("src") == s and r.get("dst") == d]
            if c:
                c = c[0]
                print(f"   R={R_:3d} {s}->{d}: {c['ms_full']:.4f} ms/call "
                      f"({c['tflops']:.1f} TF/s) = {full['ms']/c['ms_full']:.3f}x production")
print("\n=== SDPA batch chunking (P6/P7) ===")
for r in fl["sdpa"]:
    if "ms" in r:
        print(f"  B={r['B']:3d} qkv={r.get('qkv_mem'):4s} bias={r.get('bias','on'):3s} "
              f"{r['ms']:.4f} ms x{r.get('chunks',1)} = {r.get('ms_full', r['ms']):.4f} ms/call")
