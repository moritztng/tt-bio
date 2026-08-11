#!/usr/bin/env python3
"""Pairformer arithmetic + byte floor at 512 aa, protenix-v2.

No device. Exact FLOP and DRAM-byte counts derived from tt_bio/tenstorrent.py and
tt_bio/protenix.py at the shapes a 512 aa protenix-v2 fold actually runs, priced against
rates already measured on this card and recorded in state/pairformer-resident-chunking.md.

The point: a floor on time is a ceiling on the fusion prize. Nothing here is a projection
of a saving -- it is a bound that no execution strategy can beat.
"""

S = 512          # tokens (the goal size)
CZ = 256         # protenix.py:1983  C_Z = 256
HD = 32          # protenix.py:1984  TRI_HEAD_DIM = 32
NH = CZ // HD    # protenix.py:1991  n_tri_heads = c_z // 32 = 8
HID = 256        # trimul hidden: _hidden = g_in.t().shape[1]//2; n_pairs=hid//32=8
BLOCKS = 48      # protenix.py:2006  48-block Pairformer stack
CYCLES = 10      # protenix.py:1982  N_CYCLES = 10
LAYERS = BLOCKS * CYCLES                      # 480
M = S * S                                     # 262144 pair rows
P = S * S * CZ * 2                            # pair tensor, bf16 = 134.218 MB

# --- rates, every one measured on this card and cited -------------------------------------
R_TRI = 40.40e12    # §62: trimul.tri_matmul at 512 aa on qb2 WITH its program config
R_BEST = 93.47e12   # §1/brief: K=256, nt=64, L1 output -- the best this card has shown
R_SPEC = 664e12/4*110/120   # vendor 664 TFLOPS BLOCKFP8 / 4 passes for bf16 HiFi4, 110 of 120 cores
BW = 277.6e9        # measured device write roof (memory tt-bio-matmul-dram-write-serialized)
BW_SPEC = 512e9     # vendor GDDR6

def mm(m, k, n):
    return 2 * m * k * n

# --- TriangleMultiplication, per invocation --------------------------------------------------
# --fast at 512 aa: _trimul_l1_max_seq()=640 so the channel loop is L1, chunk=32, n_pairs=8,
# group=1 (large_seq False). H=512 < SEQ_LEN_MORE_CHUNKING=1536 so the simple tail runs.
tm_inproj = mm(M, CZ, 2 * 2 * HID)            # g_in|p_in, each CZ->2*HID, over all 8 chunks
tm_tri    = HID * (2 * S * S * S)             # 256 batched SxSxS, a_chunk @ b_chunk
tm_out    = 2 * mm(M, CZ, CZ)                 # p_out and g_out
TM_FLOP   = tm_inproj + tm_tri + tm_out

# DRAM today: norm 2P, in-projection re-reads x_norm_in once per chunk (8P), chunk clones to
# DRAM (P), concat (2P), tail norm (2P) + two out-projections (2P read, ~P written) + gate.
TM_BYTES_NOW   = (2 + 8 + 1 + 2 + 2 + 2 + 1 + 1) * P
# DRAM for a perfectly fused trimul: read the pair tensor once, write the result once.
TM_BYTES_FUSED = 2 * P

# --- TriangleAttention, per invocation ---------------------------------------------------------
# need_chunk is False at 512 aa (S=512 < SEQ_LEN_MORE_CHUNKING=1536), so the simple path runs.
ta_qkv  = mm(M, CZ, 3 * NH * HD)
ta_bias = mm(M, CZ, NH)
ta_sdpa = S * NH * (mm(S, HD, S) + mm(S, S, HD))   # QK^T then AV, per row per head
ta_out  = mm(M, NH * HD, CZ)
TA_FLOP = ta_qkv + ta_bias + ta_sdpa + ta_out
TA_BYTES_NOW   = 8 * P     # transpose, norm, qkv, bias, sdpa in/out, gate, out-proj
TA_BYTES_FUSED = 2 * P

PER_LAYER_FLOP = 2 * TM_FLOP + 2 * TA_FLOP     # starting + ending of each
BODY_FLOP = PER_LAYER_FLOP * LAYERS

# --- the measured basis (§63 g1, qb2, ttnn 0.68.0, CIF 98c33a481fa1fd27) -------------------------
FOLD = 79.978
W_TM, W_TA = 30.9444, 22.7278          # body wall, seconds
W_MM_TM, W_MM_TA = 10.133, 7.585       # §62a matmul ledger inside those bodies
TARGET = 48.904                        # 4x of the H200's 12.226 s

def band(x):
    return f"{x:,.3f}"

print("=" * 78)
print("Pairformer floor, protenix-v2 @ 512 aa, 480 layers x 2 of each body")
print("=" * 78)
print(f"trimul   FLOP/invocation : {TM_FLOP:.4e}  (inproj {tm_inproj:.3e}, tri {tm_tri:.3e}, out {tm_out:.3e})")
print(f"tri-att  FLOP/invocation : {TA_FLOP:.4e}  (qkv {ta_qkv:.3e}, sdpa {ta_sdpa:.3e}, out {ta_out:.3e})")
print(f"both bodies, per fold    : {BODY_FLOP:.4e} FLOP")
print()
print(f"trimul DRAM today        : {TM_BYTES_NOW/1e6:,.1f} MB/invocation   fused floor {TM_BYTES_FUSED/1e6:,.1f} MB")
print(f"  of which the 8x re-read: {8*P/1e6:,.1f} MB  -> {8*P*2*LAYERS/BW:,.2f} s/fold at {BW/1e9:.1f} GB/s")
print(f"  (§55 measured this independently at ~3 s/fold -- byte model reproduces it)")
print()
print("--- what the two bodies cost today ---")
print(f"wall                     : {W_TM+W_TA:,.3f} s  ({(W_TM+W_TA)/FOLD*100:.1f} % of the {FOLD} s fold)")
print(f"effective rate           : {BODY_FLOP/(W_TM+W_TA)/1e12:,.2f} TFLOP/s")
print(f"matmul ledger inside them: {W_MM_TM+W_MM_TA:,.3f} s -> implies {BODY_FLOP/(W_MM_TM+W_MM_TA)/1e12:,.1f} TFLOP/s across the matmuls")
print(f"non-arithmetic residual  : {W_TM+W_TA-W_MM_TM-W_MM_TA:,.3f} s "
      f"({(W_TM+W_TA-W_MM_TM-W_MM_TA)/(W_TM+W_TA)*100:.0f} % of the two bodies)")
print()
print("--- the floor, at three roofs ---")
for name, r in (("tri_matmul measured 40.40", R_TRI), ("card best 93.47", R_BEST), ("vendor bf16 HiFi4 110c", R_SPEC)):
    t = BODY_FLOP / r
    byt = (2*TM_BYTES_FUSED + 2*TA_BYTES_FUSED) * LAYERS / BW
    floor = max(t, byt)
    fold_floor = FOLD - (W_TM + W_TA) + floor
    print(f"  {name:26s}: arith {band(t)} s, bytes {band(byt)} s -> floor {band(floor)} s"
          f" -> fold {band(fold_floor)} s ({FOLD/fold_floor:.2f}x, target {TARGET})")
print()
print("--- the requirement, stated as what must be deleted ---")
need = FOLD - TARGET
budget = (W_TM + W_TA) - need
nonarith = W_TM + W_TA - W_MM_TM - W_MM_TA
print(f"fold must lose           : {band(need)} s")
print(f"two-body budget after    : {band(budget)} s   (everything outside them held fixed)")
print(f"matmuls alone cost       : {band(W_MM_TM+W_MM_TA)} s  -> leaves {band(budget-W_MM_TM-W_MM_TA)} s for all non-arithmetic")
print(f"non-arithmetic today     : {band(nonarith)} s")
print(f"=> must delete           : {(1-(budget-W_MM_TM-W_MM_TA)/nonarith)*100:.1f} % of the non-arithmetic time")
print(f"   (or, if fusion also lifts the matmuls to {R_TRI/1e12:.2f} TFLOP/s = {band(BODY_FLOP/R_TRI)} s: "
      f"{(1-(budget-BODY_FLOP/R_TRI)/nonarith)*100:.1f} %)")
print()
print("--- is that inside the byte roof? ---")
na_bytes = (2*(TM_BYTES_NOW-TM_BYTES_FUSED) + 2*(TA_BYTES_NOW-TA_BYTES_FUSED)) * LAYERS
print(f"non-arithmetic DRAM today: {na_bytes/1e9:,.1f} GB -> {band(na_bytes/BW)} s at the {BW/1e9:.1f} GB/s roof")
print(f"non-arithmetic wall      : {band(nonarith)} s -> running at {na_bytes/BW/nonarith*100:.1f} % of its own byte roof")
