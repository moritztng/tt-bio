#!/usr/bin/env python3
"""p63 -- the irreducible-traffic census for RFD3 at the page fixture (9q6y, 685 tokens,
6051 atoms, 200 timesteps, batch 1).

Arithmetic only, no device time. Two byte counts per op per step:

  ACTUAL      what the shipped op sequence moves across DRAM, from the shapes in
              tt_bio/rfd3/model.py.
  IRREDUCIBLE what a perfectly fused implementation of the same mathematics must move:
              every weight once per step, every region input once, every region output
              once, and zero for any value that is recomputable or that fits in L1 for
              the span it is live. Every line carries its assumption.

The cost model is three MEASURED roofs plus one fitted constant:

  t_op = read_MB/390.0 + write_MB/W_DRAM + gflop/102.02 + C_OP

  390.0 GB/s   DRAM read roof, measured on this chip 2026-08-09 (perfwar-rfd3-esmfold2-sites)
  163.9 GB/s   DRAM write rate a matmul's out-CB drain actually reaches: the drain is issued
               by BRISC alone (Tracy profile, tt-bio-dram-write-serialization). The 269.6 GB/s
               write roof needs both data-movement RISCs and no ttnn op splits the drain today.
  102.02 TFLOP/s  bf16 HiFi4 matmul roof, same measurement session.
  C_OP         per-op cost that does NOT scale with bytes, fitted below against the two
               regions whose unsynced wall is trusted (p49 v2, p46 calibrated).

Units: MB/(GB/s) = ms and GFLOP/(TFLOP/s) = ms, so every term is already ms.
"""
import json, sys

READ_ROOF = 390.0        # GB/s
WRITE_DRAIN = 163.9      # GB/s, measured single-RISC out-CB drain
WRITE_ROOF = 269.6       # GB/s, both RISCs; used only for the irreducible-floor variant
COMPUTE_ROOF = 102.02    # TFLOP/s bf16 HiFi4
L1_AGG_GB_S = 4216.0     # aggregate L1, from the task brief's own figure

I, IP = 685, 704
L, LP = 6051, 6080
TILE = 32


def c32(c):
    return -(-c // TILE) * TILE


def PAIR(c, dt=2):
    """[1, I, IP, c] tiled on the last two dims; dim 1 is not tiled."""
    return I * IP * c32(c) * dt / 1e6


def HEAD(h, dt=2):
    """[1, h, IP, IP]."""
    return h * IP * IP * dt / 1e6


def TOK(c, dt=2):
    return IP * c32(c) * dt / 1e6


def ATOMP(c, dt=2):
    """sparse atom pair [1, L, 128, c]."""
    return L * 128 * c32(c) * dt / 1e6


def ATOM(c, dt=2):
    return LP * c32(c) * dt / 1e6


def mm_gflop(m, k, n):
    return 2 * m * k * n / 1e9


U = PAIR(128)                      # 123.47 MB, the pair tensor
PAIRROWS = I * IP                  # 482240 positions of 128 channels

OPS = []   # (region, label, n_per_step, read_MB, write_MB, gflop, irr_read, irr_write, irr_gflop, note)


def op(region, label, n, r, w, g, ir, iw, ig, note, wkind="copy"):
    """wkind: "mm" -> the write is a matmul out-CB drain, which one RISC issues at 163.9 GB/s
    (Tracy profile, tt-bio-dram-write-serialization). "copy" -> an eltwise/movement op, which
    reaches the 269.6 GB/s write roof (p52 concat 356 GB/s r+w, p60 slice 302 GB/s r+w)."""
    OPS.append(dict(region=region, label=label, n=n, r=r * n, w=w * n, g=g * n,
                    ir=ir * n, iw=iw * n, ig=ig * n, note=note, wkind=wkind))


# ---------------------------------------------------------------- transition helper
def transition(region, label, n, unit, hidden, chan, note_extra=""):
    """One RFD3 Transition (model.py:514): rms_norm, fc1(silu), fc2, multiply, fc3, residual add.

    `unit` is the bytes of one [.., chan] tensor; the hidden tensors are unit*hidden/chan.
    ACTUAL: every one of the six ops round-trips DRAM.
    IRREDUCIBLE: read the input once, write the output once. a/b/m are dead the instant
    fc3 consumes them and a row block of them fits L1, so they are zero.
    """
    h = unit * hidden / c32(chan)
    m = PAIRROWS if unit >= 100 else IP
    fl = 3 * mm_gflop(m, c32(chan), hidden)
    op(region, f"{label}: rms_norm", n, unit, unit, 0, 0, 0, 0, "folded into the fused block")
    op(region, f"{label}: fc1 silu [{chan}->{hidden}]", n, unit, h, fl / 3, 0, 0, fl / 3, "compute stays", "mm")
    op(region, f"{label}: fc2 [{chan}->{hidden}]", n, unit, h, fl / 3, 0, 0, fl / 3, "compute stays", "mm")
    op(region, f"{label}: multiply a*b", n, 2 * h, h, 0, 0, 0, 0, "L1-resident intermediate")
    op(region, f"{label}: fc3 [{hidden}->{chan}]", n, h, unit, fl / 3, 0, 0, fl / 3, "compute stays", "mm")
    op(region, f"{label}: residual add", n, 2 * unit, unit, 0, unit, unit, 0,
       "the ONE read and ONE write the whole Transition needs" + note_extra)


# ================================================================ R1 token encoder
# 2 calls/step (N_RECYCLE=2). p46 calibrated: 249.98 ms/step unsynced, oversync 1.00x.
# The aligned concat (P3.19) landed after p46, so the shipped region is ~206 ms/step today;
# every non-concat row below is unchanged by that lever.
R1 = "token encoder"
transition(R1, "transition_1 x2 [s,384,H=768]", 4, TOK(384), 768, 384)
op(R1, "distogram+self one-hot (2x ttnn.embedding)", 2,
   2 * 1.93, PAIR(192), 0,
   0, 0, 0,
   "IRREDUCIBLE 0: a one-hot of a bin index is a row gather from the weight; the 192-wide "
   "tensor never has to exist. The two int32 bin maps are the real input, 1.93 MB each.")
op(R1, "concat [z|d|dself] -> PAIR(320)", 2,
   U + PAIR(192), PAIR(320), 0, 0, 0, 0, "IRREDUCIBLE 0: fused away")
op(R1, "slice PAIR(320) -> PAIR(258)", 2,
   PAIR(320), PAIR(288), 0, 0, 0, 0, "IRREDUCIBLE 0: fused away")
op(R1, "rms_norm(258)", 2, PAIR(288), PAIR(288), 0, 0, 0, 0,
   "IRREDUCIBLE 0: rms over 258 channels = rms over z's 128 plus the constant 2 the two "
   "one-hots contribute, so it is computable from z alone")
op(R1, "process_z linear 258->128", 2,
   PAIR(288), U, mm_gflop(PAIRROWS, 288, 128),
   U + 2 * 1.93, U, mm_gflop(PAIRROWS, 288, 128),
   "IRREDUCIBLE: read z + the two bin maps once, write z once. The one-hot half of the "
   "matmul is two weight-row gathers, so its FLOPs are kept as an upper bound.", "mm")
transition(R1, "transition_2 x2 [z,128,H=256]", 4, U, 256, 128)
transition(R1, "pairformer z_transition x2 [z,128,H=512]", 4, U, 512, 128)
# PairformerAttention x2 x 2 recycles
op(R1, "pf attn: rms_norm(z)", 4, U, U, 0, 0, 0, 0, "fused into the pair pass")
op(R1, "pf attn: to_b linear 128->16", 4, U, PAIR(32), mm_gflop(PAIRROWS, 128, 32),
   0, PAIR(32) / 2, mm_gflop(PAIRROWS, 128, 16),
   "IRREDUCIBLE: z is already being read by the fused pair pass, so only the 16-wide bias "
   "write is charged, unpadded", "mm")
op(R1, "pf attn: permute bias -> [1,16,I,I]", 4, PAIR(32), HEAD(16), 0, 0, 0, 0,
   "IRREDUCIBLE 0: a fused kernel emits the head-major layout directly")
op(R1, "pf attn: q/k/v/g/o + 2 rms_norm on TOK(384)", 4,
   9 * TOK(384), 8 * TOK(384), 5 * mm_gflop(IP, 384, 384),
   TOK(384), TOK(384), 5 * mm_gflop(IP, 384, 384), "token-vector work, negligible bytes")
op(R1, "pf attn: scores + fp32 softmax chain (6 ops)", 4,
   HEAD(16) + HEAD(16) + HEAD(16, 4) + 2 * HEAD(16, 4) + HEAD(16, 4) + HEAD(16, 4),
   HEAD(16) + HEAD(16, 4) + HEAD(16, 4) + HEAD(16, 4) + HEAD(16),
   2 * mm_gflop(16 * IP, 24, IP) / 16 * 16,
   HEAD(16), HEAD(16), 2 * mm_gflop(16 * IP, 24, IP) / 16 * 16,
   "IRREDUCIBLE: scores written once bf16, read once by a fused softmax+PV; no fp32 copy")
transition(R1, "pf s_transition x2 [s,384,H=1536]", 4, TOK(384), 1536, 384)

# ================================================================ R2 token DiT
# 36 blocks/step (18 blocks x 2 recycles). p49 v2: 133.967 ms/step unsynced, 183.088 synced.
R2 = "token DiT"
op(R2, "pair_bias linear 128->16 (model.py:1492)", 36,
   U, PAIR(32), mm_gflop(PAIRROWS, 128, 32),
   U / 18, PAIR(32) / 18, mm_gflop(PAIRROWS, 128, 32),
   "IRREDUCIBLE: z is invariant across the 18 blocks and differs only across the 2 recycles, "
   "so 2 reads, not 36. Charged as 1/18 of the shipped read.", "mm")
op(R2, "permute pair_bias -> [1,16,I,I]", 36, PAIR(32), HEAD(16), 0, 0, 0, 0,
   "IRREDUCIBLE 0: emit head-major from the fused projection")
op(R2, "add(pair_bias, additive_mask)", 36, 2 * HEAD(16), HEAD(16), 0, 0, 0, 0, "fused")
op(R2, "pad x3 (kk, vv, bias)", 108, 2.0, 2.0, 0, 0, 0, 0,
   "IRREDUCIBLE 0: 704 is already tile-aligned, this is defensive")
op(R2, "typecast bias -> fp32", 36, HEAD(16), HEAD(16, 4), 0, 0, 0, 0, "fused")
op(R2, "matmul scores qq@kt", 36, 2 * TOK(768), HEAD(16), mm_gflop(16 * IP, 48, IP),
   0, HEAD(16), mm_gflop(16 * IP, 48, IP), "scores written once")
op(R2, "typecast scores -> fp32", 36, HEAD(16), HEAD(16, 4), 0, 0, 0, 0, "fused")
op(R2, "add scores + bias_f (fp32)", 36, 2 * HEAD(16, 4), HEAD(16, 4), 0, 0, 0, 0, "fused")
op(R2, "softmax fp32", 36, HEAD(16, 4), HEAD(16, 4), 0, HEAD(16), HEAD(16), 0,
   "IRREDUCIBLE: one bf16 read and one bf16 write; the fp32 residency is an implementation choice")
op(R2, "typecast attn -> bf16", 36, HEAD(16, 4), HEAD(16), 0, 0, 0, 0, "fused")
op(R2, "attn@vv + merge_heads", 36, HEAD(16) + TOK(768), 2 * TOK(768),
   mm_gflop(16 * IP, IP, 48), HEAD(16), TOK(768), mm_gflop(16 * IP, IP, 48), "")
op(R2, "12 linears on TOK(768)/TOK(384)", 432,
   TOK(768) + 1.2, TOK(768), mm_gflop(IP, 768, 768),
   0, 0, mm_gflop(IP, 768, 768),
   "IRREDUCIBLE bytes ~0: a whole DiT block's token-vector state is 1.08 MB and lives in L1 "
   "for the block; only the weights must be read, charged in the weight line")
op(R2, "reshape x3 + permute x3 (head split)", 216, TOK(768), TOK(768), 0, 0, 0, 0,
   "IRREDUCIBLE 0: a fused qkv projection emits head-major")
op(R2, "rms_norm/sigmoid/multiply/add on TOK (the rest)", 900,
   TOK(768), TOK(768), 0, 0, 0, 0, "IRREDUCIBLE 0: L1-resident block state")
op(R2, "DiT weights, read once per recycle", 2, 276.0, 0, 0, 276.0, 0, 0,
   "18 blocks x (5x768x768 + 3x768x1536 + 4x384x768 + 128x16) params at bf16")

# ================================================================ R3 atom decoder
# 2 calls/step, p49 v2 102.766 ms/step unsynced. 3 RFD3AtomBlocks at c_a=128, n_head=4,
# sparse k=128 keys, plus 3 upcast GCAs and 1 downcast GCA.
R3 = "atom decoder"
op(R3, "sparse pair bias [1,L,128,32]@[32,32]", 6, ATOMP(32), ATOMP(32),
   mm_gflop(L * 128, 32, 32), ATOMP(16) / 3, ATOMP(16) / 3, mm_gflop(L * 128, 32, 16),
   "IRREDUCIBLE: the gathered pair features are the same for all 3 blocks in a call "
   "(the _sparse_pair_bias cache already exploits this), so 1 read per call not 3")
op(R3, "softmax fp32 [1,4,LP,128]", 6, 4 * ATOM(128, 4), 4 * ATOM(128, 4), 0,
   LP * 4 * 128 * 2 / 1e6, LP * 4 * 128 * 2 / 1e6, 0,
   "IRREDUCIBLE: bf16 in, bf16 out, fused with the PV matmul")
op(R3, "typecast around the softmax (x2)", 12, 4 * LP * 128 * 2 / 1e6, 4 * LP * 128 * 4 / 1e6, 0,
   0, 0, 0, "fused")
op(R3, "scores + PV matmuls", 12, ATOM(128) / 2, ATOM(128) / 2, mm_gflop(4 * LP, 128, 32),
   0, 0, mm_gflop(4 * LP, 128, 32), "")
op(R3, "atom-block linears + transitions (c=128, H=256)", 138,
   ATOM(128), ATOM(128), mm_gflop(LP, 128, 128), 0, 0, mm_gflop(LP, 128, 128),
   "IRREDUCIBLE bytes 0: the atom state is 1.56 MB and stays in L1 across the block")
op(R3, "reshape/permute/pack/unpack/gather (the rest)", 450,
   ATOM(128), ATOM(128), 0, 0, 0, 0, "IRREDUCIBLE 0: layout, not mathematics")
op(R3, "atom-path weights, once per call", 2, 24.0, 0, 0, 24.0, 0, 0, "")

# ================================================================ R4 atom encoder
R4 = "atom encoder"
op(R4, "sparse pair bias", 3, ATOMP(32), ATOMP(32), mm_gflop(L * 128, 32, 32),
   ATOMP(16) / 3, ATOMP(16) / 3, mm_gflop(L * 128, 32, 16), "same as the decoder")
op(R4, "softmax fp32 [1,4,LP,128]", 3, 4 * ATOM(128, 4), 4 * ATOM(128, 4), 0,
   LP * 4 * 128 * 2 / 1e6, LP * 4 * 128 * 2 / 1e6, 0, "")
op(R4, "typecast around the softmax (x2)", 6, 4 * LP * 128 * 2 / 1e6, 4 * LP * 128 * 4 / 1e6, 0,
   0, 0, 0, "fused")
op(R4, "atom-block linears + transitions", 69, ATOM(128), ATOM(128),
   mm_gflop(LP, 128, 128), 0, 0, mm_gflop(LP, 128, 128), "")
op(R4, "embedding + to_layout + the rest", 85, ATOM(128), ATOM(128), 0, 0, 0, 0, "")
op(R4, "atom-path weights, once per step", 1, 24.0, 0, 0, 24.0, 0, 0, "")

# ================================================================ report
MEASURED = {                # ms/step, unsynced region wall
    R1: 206.0,              # p46 calibrated 249.98 MINUS the aligned-concat win (P3.19, 49.5 ms/step)
    R2: 133.967,            # p49 v2
    R3: 102.766,            # p49 v2
    R4: 49.252,             # p49 v2
}
NOPS = {R1: None, R2: 2052, R3: 626, R4: 163}   # p49 v2 op counts; R1 computed below


def roofms(r, w, g, wroof=WRITE_ROOF):
    return r / READ_ROOF + w / wroof + g / COMPUTE_ROOF


def op_roofms(o, key="a"):
    r, w, g = ((o["r"], o["w"], o["g"]) if key == "a" else (o["ir"], o["iw"], o["ig"]))
    wroof = WRITE_DRAIN if o["wkind"] == "mm" else WRITE_ROOF
    return r / READ_ROOF + w / wroof + g / COMPUTE_ROOF


rows = {}
for o in OPS:
    d = rows.setdefault(o["region"], dict(r=0, w=0, g=0, ir=0, iw=0, ig=0, n=0, ta=0, ti=0))
    for k in ("r", "w", "g", "ir", "iw", "ig", "n"):
        d[k] += o[k]
    d["ta"] += op_roofms(o, "a")
    d["ti"] += op_roofms(o, "i")

print("=" * 108)
print("PHASE 0 -- IRREDUCIBLE-TRAFFIC CENSUS, RFD3 page fixture (685 tok / 6051 atoms), per timestep")
print("=" * 108)
hdr = f"{'region':<16}{'ops':>6}{'act R GB':>10}{'act W GB':>10}{'irr R GB':>10}{'irr W GB':>10}{'ratio':>8}{'act roof':>10}{'irr roof':>10}{'measured':>10}"
print(hdr)
print("-" * 108)
tot = dict(r=0, w=0, g=0, ir=0, iw=0, ig=0, n=0, meas=0)
for reg in (R1, R2, R3, R4):
    d = rows[reg]
    ratio = (d["r"] + d["w"]) / max(1e-9, d["ir"] + d["iw"])
    print(f"{reg:<16}{d['n']:>6.0f}{d['r']/1000:>10.2f}{d['w']/1000:>10.2f}"
          f"{d['ir']/1000:>10.2f}{d['iw']/1000:>10.2f}{ratio:>8.2f}"
          f"{d['ta']:>10.1f}{d['ti']:>10.1f}"
          f"{MEASURED[reg]:>10.1f}")
    for k in ("r", "w", "g", "ir", "iw", "ig", "n"):
        tot[k] += d[k]
    tot["ta"] = tot.get("ta", 0) + d["ta"]
    tot["ti"] = tot.get("ti", 0) + d["ti"]
    tot["meas"] += MEASURED[reg]
print("-" * 108)
ratio = (tot["r"] + tot["w"]) / (tot["ir"] + tot["iw"])
act_roof = tot["ta"]
irr_roof = tot["ti"]
irr_roof_opt = roofms(tot["ir"], tot["iw"], tot["ig"], wroof=WRITE_ROOF)
print(f"{'STEP':<16}{tot['n']:>6.0f}{tot['r']/1000:>10.2f}{tot['w']/1000:>10.2f}"
      f"{tot['ir']/1000:>10.2f}{tot['iw']/1000:>10.2f}{ratio:>8.2f}"
      f"{act_roof:>10.1f}{irr_roof:>10.1f}{tot['meas']:>10.1f}")
print()
print(f"actual traffic          {(tot['r']+tot['w'])/1000:8.2f} GB/step   "
      f"({tot['r']/1000:.2f} read + {tot['w']/1000:.2f} write)")
print(f"irreducible traffic     {(tot['ir']+tot['iw'])/1000:8.2f} GB/step   "
      f"({tot['ir']/1000:.2f} read + {tot['iw']/1000:.2f} write)")
print(f"ACTUAL / IRREDUCIBLE    {ratio:8.2f} x")
print()
print(f"compute, unavoidable    {tot['g']/1000:8.3f} TFLOP/step -> {tot['g']/COMPUTE_ROOF:6.1f} ms/step at 102.02 TFLOP/s")
print(f"roof time of ACTUAL     {act_roof:8.1f} ms/step  (read 390.0, write {WRITE_DRAIN} single-RISC)")
print(f"roof time of IRREDUCIBLE{irr_roof:8.1f} ms/step  (same roofs)")
print(f"  ... at the 269.6 write roof (both RISCs): {irr_roof_opt:.1f} ms/step")
print(f"measured exposed device       451.0 ms/step  (p57 clean ledger, 450.957)")
print(f"4x-at-b8 device budget        130.0 ms/step  (259.5 step - 129.5 host+dispatch)")
print()
# fitted per-op constant on the two regions whose byte model is most complete
for reg in (R2, R3, R4):
    d = rows[reg]
    n = NOPS[reg]
    resid = MEASURED[reg] - d["ta"]
    print(f"C_OP fit, {reg:<14} measured {MEASURED[reg]:7.1f} - roof {d['ta']:6.1f}"
          f" = {resid:7.1f} ms over {n} ops = {resid*1000/n:5.1f} us/op")
d = rows[R1]
n = sum(o["n"] for o in OPS if o["region"] == R1)
resid = MEASURED[R1] - d["ta"]
print(f"C_OP fit, {R1:<14} measured {MEASURED[R1]:7.1f} - roof {d['ta']:6.1f}"
      f" = {resid:7.1f} ms over {n:.0f} ops = {resid*1000/n:5.1f} us/op")

print()
print("=" * 108)
print("THE TEN LARGEST (actual - irreducible) SITES, by bytes per step")
print("=" * 108)
sites = {}
for o in OPS:
    key = o["label"].split(":")[0] if ":" in o["label"] else o["label"]
    key = f"[{o['region']}] {key}"
    s = sites.setdefault(key, dict(d=0, act=0, irr=0, g=0, n=0))
    s["d"] += (o["r"] + o["w"]) - (o["ir"] + o["iw"])
    s["act"] += o["r"] + o["w"]
    s["irr"] += o["ir"] + o["iw"]
    s["g"] += o["g"]
    s["n"] += o["n"]
print(f"{'site':<58}{'ops':>6}{'act GB':>9}{'irr GB':>9}{'delta GB':>10}{'d roof ms':>11}")
print("-" * 108)
for k, s in sorted(sites.items(), key=lambda kv: -kv[1]["d"])[:12]:
    print(f"{k:<58}{s['n']:>6.0f}{s['act']/1000:>9.2f}{s['irr']/1000:>9.2f}{s['d']/1000:>10.2f}"
          f"{s['d']/1000*1000/((READ_ROOF+WRITE_DRAIN)/2):>11.1f}")

out = dict(fixture="9q6y chain A, 685 tokens, 6051 atoms, 200 timesteps, batch 1",
           roofs=dict(read_GB_s=READ_ROOF, write_drain_GB_s=WRITE_DRAIN,
                      write_roof_GB_s=WRITE_ROOF, compute_TFLOP_s=COMPUTE_ROOF),
           per_step=dict(
               actual_read_GB=tot["r"] / 1000, actual_write_GB=tot["w"] / 1000,
               irreducible_read_GB=tot["ir"] / 1000, irreducible_write_GB=tot["iw"] / 1000,
               ratio=ratio, actual_roof_ms=act_roof, irreducible_roof_ms=irr_roof,
               irreducible_roof_ms_both_risc=irr_roof_opt,
               compute_TFLOP=tot["g"] / 1000, compute_roof_ms=tot["g"] / COMPUTE_ROOF,
               measured_region_wall_ms=tot["meas"], device_ops=tot["n"]),
           regions={reg: dict(ops=rows[reg]["n"],
                              actual_GB=(rows[reg]["r"] + rows[reg]["w"]) / 1000,
                              irreducible_GB=(rows[reg]["ir"] + rows[reg]["iw"]) / 1000,
                              ratio=(rows[reg]["r"] + rows[reg]["w"]) / max(1e-9, rows[reg]["ir"] + rows[reg]["iw"]),
                              actual_roof_ms=rows[reg]["ta"],
                              irreducible_roof_ms=rows[reg]["ti"],
                              measured_ms=MEASURED[reg]) for reg in (R1, R2, R3, R4)},
           sites={k: dict(ops=v["n"], actual_GB=v["act"] / 1000, irreducible_GB=v["irr"] / 1000,
                          delta_GB=v["d"] / 1000) for k, v in sites.items()},
           ops=[dict(region=o["region"], label=o["label"], n=o["n"], wkind=o["wkind"], read_MB=o["r"],
                     write_MB=o["w"], gflop=o["g"], irr_read_MB=o["ir"], irr_write_MB=o["iw"],
                     assumption=o["note"]) for o in OPS])
if len(sys.argv) > 1:
    with open(sys.argv[1], "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {sys.argv[1]}")


# ================================================================ PHASE 1 prediction
# The largest (actual - irreducible) site is the pair-shaped Transition (model.py:514) --
# 4x H=256 (transition_2) + 4x H=512 (pairformer z_transition) per step, 23.70 GB/step of
# traffic against 1.98 GB irreducible. The route is the one tt_bio ALREADY ships for the
# Boltz-2/Protenix Transition (tenstorrent.py:3934): row-chunk the pair tensor and keep
# x_norm / a / b / m in L1 so only the input read and the output write reach DRAM.
print()
print("=" * 108)
print("PHASE 1 PREDICTION -- L1-resident row-chunked pair Transition (model.py:514)")
print("=" * 108)

L1_PER_CORE = 1_461_760          # allocator's bytes per L1 bank on this p150 (perfwar W5)
L1_BANKS = 140
L1_AGG = L1_PER_CORE * L1_BANKS / 1e6      # MB


def blocked_transition_ms(hidden, h_rows, c_op_us, tflops):
    """One pair Transition, row-chunked on dim 1 (untiled, so the slice has no sub-tile cliff).

    DRAM: read the input once, write the output once, plus the weights once per chunk.
    L1:   x_norm/a/b/m live and die inside the chunk, priced at the aggregate L1 rate.
    ops:  ceil(685/h) chunks x 6 ops + 1 closing concat.
    """
    n_blk = -(-I // h_rows)
    u_blk = U * h_rows / I
    h_blk = u_blk * hidden / 128
    dram_r = U + n_blk * (3 * 128 * hidden * 2 / 1e6)
    dram_w = U
    l1_bytes = n_blk * (2 * u_blk + 6 * h_blk + 2 * u_blk)
    gflop = 3 * mm_gflop(PAIRROWS, 128, hidden)
    ops = n_blk * 6 + 1
    t = dram_r / READ_ROOF + dram_w / WRITE_ROOF + l1_bytes / L1_AGG_GB_S \
        + gflop / tflops + ops * c_op_us / 1000.0
    peak_l1 = 2 * u_blk + 2 * h_blk        # x_norm + a + b live together (m reuses a/b)
    return t, ops, peak_l1, n_blk


print(f"aggregate L1 on this p150: {L1_AGG:.1f} MB ({L1_BANKS} banks x {L1_PER_CORE} B)")
print()
print(f"{'H':>5}{'h rows':>8}{'chunks':>8}{'peak L1 MB':>12}{'fits':>6}{'ops':>6}"
      f"{'ms @102TF':>11}{'ms @30TF':>10}{'ms @15TF':>10}")
print("-" * 108)
for hidden in (256, 512):
    for h_rows in (16, 32, 64, 96, 128):
        row = []
        for tf in (COMPUTE_ROOF, 30.0, 15.0):
            t, ops, peak, n_blk = blocked_transition_ms(hidden, h_rows, 30.0, tf)
            row.append(t)
        fits = "yes" if peak < 0.75 * L1_AGG else "NO"
        print(f"{hidden:>5}{h_rows:>8}{n_blk:>8}{peak:>12.1f}{fits:>6}{ops:>6}"
              f"{row[0]:>11.2f}{row[1]:>10.2f}{row[2]:>10.2f}")

print()
print("shipped cost of the same eight calls, from the additive roof model and the measurements:")
ship256 = sum(op_roofms(o, "a") for o in OPS if "transition_2 x2" in o["label"]) / 4
ship512 = sum(op_roofms(o, "a") for o in OPS if "z_transition x2" in o["label"]) / 4
print(f"  H=256 model {ship256:.2f} ms/call   vs measured 14.876 ms/call (p46, 59.504 ms/step / 4)")
print(f"  H=512 model {ship512:.2f} ms/call   vs measured ~18.9 ms/call (inside the 25.39 ms/call")
print(f"                                        pairformer block wall; the screen's arm A splits it)")
SHIP_STEP = 59.504 + 4 * 18.9
print(f"  shipped, eight calls per step: {SHIP_STEP:.1f} ms/step")
print()
for tf, tag in ((COMPUTE_ROOF, "optimistic: L1-fed matmul at the roof"),
                (30.0, "central: L1-fed matmul at 30 TFLOP/s"),
                (15.0, "pessimistic: k_tiles=4 pack amplification holds, 15 TFLOP/s")):
    for c_op in (30.0, 80.0):
        t256 = blocked_transition_ms(256, 64, c_op, tf)[0]
        t512 = blocked_transition_ms(512, 32, c_op, tf)[0]
        new = 4 * t256 + 4 * t512
        print(f"  {tag:<52} C_OP={c_op:>4.0f}us -> {new:6.1f} ms/step, "
              f"saves {SHIP_STEP-new:6.1f} ms/step = {(SHIP_STEP-new)*200/1000:5.2f} s/design")
