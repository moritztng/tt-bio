#!/usr/bin/env python3
"""Per-op DRAM byte attribution of OUR Boltz-2 pairformer block at 512 aa, counted from source.

The campaign's fusion bet has to be ranked against our own block, not against BioIR's. P0
(`b2x-baseline-attrib`) is producing that attribution on a card; this is the card-free version:
every op the block issues at 512 aa, read line by line off `tt_bio/tenstorrent.py` on the path
the shipped configuration actually takes, priced in Z (one bf16 pair tensor, 67.11 MB).

Same counting rule `state/bioir-roofline/_scripts/bioir_block_bytes.py` applied to BioIR, so the
two sides stay comparable: a fused op counts once, inputs read once and outputs written once.
L1 traffic is NOT counted -- the 12.170 GB the capture reports is `dram_read + dram_write` and it
reports `l1_write` separately (capture_rows.json, fold_bytes_512_p300c_qb2).

The path at 512 aa, with the constants that pick it:
  _trimul_l1_max_seq() = TRIANGLE_MULT_L1_MAX_SEQ = 352  -> 512 > 352, so DRAM, not L1.
  TRIMUL_IN_NORM_ROWBLOCK_BYTES = 3 GiB  -> z is 67.11 MB, so the input norm is whole-tensor.
  SEQ_LEN_MORE_CHUNKING = 1536           -> 512 < 1536, so the output tail is whole-tensor and
                                            triangle attention does not row-chunk.
  _trimul_chunk_size -> 32, n_pairs = 128/32 = 4, _trimul_inproj_group -> 4 (4*32*4*512^2*2 =
                        268 MB, inside the 1 GiB _TRIMUL_INPROJ_FUSED_BYTES cap), so the channel
                        loop runs ONCE over all 128 hidden channels.
  PAIRFORMER_PAD_MULTIPLE = bucket_multiple("boltz2") = 32, seq_len 512 -> seq_pad = 0.
  gated_move is False for boltz2 (no call site passes it), so E6 is off.
  F1_BLOCK_KEYS = {(8,8)} and boltz2's tail weight is [128,128] -> key (4,4) -> F1 declines.
"""
import json
from pathlib import Path

N, C_Z = 512, 128
Z = N * N * C_Z * 2                       # 67 108 864 B

# (name, read_Z, write_Z). One row per op the block issues.
TRIMUL_IN = [
    ("layer_norm(z) -> x_norm_in", 1, 1),
    ("_in_proj_matmul: x_norm_in @ [128,512] -> gp 4Z (mm_dualnoc)", 1, 4),
    ("ttnn.chunk(gp, 4, -1) -> g_a|g_b|p_a|p_b", 4, 4),
    ("multiply_(p_a, g_a, SIGMOID) -> a", 2, 1),
    ("multiply_(p_b, g_b, SIGMOID) -> b", 2, 1),
    ("multiply_(a, mask_u)", 1, 1),
    ("_channel_move(a) (0,3,1,2), reblock_permute", 1, 1),
    ("_channel_move(b) (0,3,1,2), reblock_permute", 1, 1),
    ("ttnn.transpose(b, -2, -1)", 1, 1),
]
TRIMUL_MID = [
    ("ttnn.matmul(a, b) -> x_chunk", 2, 1),
    ("_channel_move_back(x_chunk), reblock_permute_back", 1, 1),
]
TRIMUL_TAIL = [
    ("layer_norm(x_chunk) -> x_norm_out", 1, 1),
    ("_trimul_out_proj(x_norm_out, Wp) -> p_out", 1, 1),
    ("_trimul_out_proj(x_norm_in, Wg) -> g_out", 1, 1),
    ("multiply_(p_out, g_out, SIGMOID)", 2, 1),
]
# Triangle attention, unchunked path. n_heads = 4, head_dim = 32, so qkv is 3*128 = 384 ch = 3Z
# and the [1,4,N,N] triangle bias is 4/128 of Z.
BIAS = 4 / C_Z
TRIATT = [
    ("_pair_transpose(z) [ending only]", 1, 1),
    ("layer_norm(z) -> x", 1, 1),
    ("_pair_proj_linear(x, Wbias) -> triangle_bias", 1, BIAS),
    ("permute(bias, (0,3,1,2))", BIAS, BIAS),
    ("add(bias, attn_mask)", BIAS, BIAS),
    ("_triatt_qkv.qkv_heads(x, Wqkv) -> q|k|v (3Z)", 1, 3),
    ("_triatt_qkv.gate_proj(x, Wg) -> g", 1, 1),
    ("_tri_att_sdpa(q,k,v,bias) -> o", 3 + BIAS, 1),
    ("multiply_(o, g, SIGMOID)", 2, 1),
    ("_triatt_qkv.out_proj(o, Wo)", 1, 1),
    ("_pair_transpose(out) [ending only]", 1, 1),
]
ENDING_ONLY = {"_pair_transpose(z) [ending only]", "_pair_transpose(out) [ending only]"}
# Transition: 32 row blocks of 16 rows, hidden 512. layer_norm / fc1 / fc2 / multiply_ / fc3 all
# run with memory_config=L1 except the last, which writes DRAM, so per block one z-row-block is
# read and one written, and the three [128,512]-ish weights are re-read per block.
W_TRANS = 3 * C_Z * 4 * C_Z * 2 * 32 / Z
TRANSITION = [
    ("layer_norm(rows) x32 blocks: read z", 1, 0),
    ("fc3 -> DRAM x32 blocks: write update", 0, 1),
    ("fc1/fc2/fc3 weights re-read per block", W_TRANS, 0),
    ("ttnn.concat(32 parts)", 1, 1),
]
RESIDUALS = [("ttnn.add_(z, z_update) x5", 2 * 5, 1 * 5)]
# transform_s=True on the trunk: AttentionPairBias projects z to a [1,16,N,N] pair bias.
S_TRACK = [("attention_pair_bias: z -> [1,16,N,N] bias", 1, 16 / C_Z)]

MEASURED_BLOCK_BYTES = 12_169_659_392        # capture_rows.json, p300c qb2 card 2
MEASURED_BLOCK_MS = 41.6326
MEASURED_MSA_BYTES = 11_808_609_792
MEASURED_MSA_MS = 39.0606
BLOCK_CALLS, MSA_CALLS = 264, 16
FOLD_BYTES = 6_453_653_463_441.067
CELL_S, CELL_DEVICE_S = 23.504, 23.122
BW_ROOF = 429.9e9


def tot(rows, drop=()):
    return sum(r + w for n, r, w in rows if n not in drop)


def main():
    trimul_start = tot(TRIMUL_IN) + tot(TRIMUL_MID) + tot(TRIMUL_TAIL)
    trimul = 2 * trimul_start                       # start and end cost the same (perm_a/perm_b swap)
    triatt = tot(TRIATT, drop=ENDING_ONLY) + tot(TRIATT)
    trans, resid, strack = tot(TRANSITION), tot(RESIDUALS), tot(S_TRACK)
    attributed = trimul + triatt + trans + resid + strack
    measured_Z = MEASURED_BLOCK_BYTES / Z

    # The boundary, and what each step of the design deletes from it, per trimul. Two ops survive
    # every step and are excluded from every replaced set: the input `layer_norm` (2Z -- the tail's
    # g_out projection reads x_norm_in, so it cannot be dropped) and `transpose(b, -2, -1)` (2Z --
    # E6 leaves the inner L,L swap outside the kernel and so does this design).
    SURVIVE = 2 + 2
    # Step 1: move the pair-mask multiply to the far side of the channel move so it stops blocking
    # `gated`, and switch E6 on for boltz2. Replaced: chunk + both gates + mask + both moves.
    # New: two gated_move calls (2Z read + 1Z write each) and the mask, still 2Z, now post-move.
    r1 = tot(TRIMUL_IN) - SURVIVE - 5               # 20Z; the in-projection is untouched here
    step1, step1_nopad = r1 - (6 + 2), r1 - 6       # mask all-ones at seq_pad = 0: drop it outright
    # Step 2: row-block the in-projection into L1 and feed the gated move from there (the `out` /
    # `row_off` mode reblock_permute_gated already implements). The projection's 4Z DRAM write and
    # the gate's 4Z DRAM read both become L1 traffic, which the 12.170 GB does not count.
    r2 = tot(TRIMUL_IN) - SURVIVE                   # 25Z, now including the in-projection
    step12, step12_nopad = r2 - (3 + 2), r2 - 3
    # Step 3: F1 (`dual_gemm_x0_x1`, already built) extended to boltz2's (4,4) block key.
    step3 = tot(TRIMUL_TAIL) - 5                    # 9Z -> LN 2Z + one fused pass 3Z

    def fold(dz_per_trimul):
        """Fold seconds and ratio for a per-trimul Z saving, at the block's own achieved rate."""
        dblock = 2 * dz_per_trimul * Z
        pf_ms = MEASURED_BLOCK_MS * (1 - dblock / MEASURED_BLOCK_BYTES)
        msa_ms = MEASURED_MSA_MS * (1 - dblock / MEASURED_MSA_BYTES)
        raw_s = (CELL_DEVICE_S - BLOCK_CALLS * MEASURED_BLOCK_MS / 1e3
                 - MSA_CALLS * MEASURED_MSA_MS / 1e3
                 + BLOCK_CALLS * pf_ms / 1e3 + MSA_CALLS * msa_ms / 1e3)
        host = CELL_S - CELL_DEVICE_S
        out = {"deleted_Z_per_trimul": dz_per_trimul,
               "deleted_GB_per_block": round(dblock / 1e9, 3),
               "block_ms_raw": round(pf_ms, 3),
               "block_ratio_raw": round(MEASURED_BLOCK_MS / pf_ms, 4),
               "fold_s_raw": round(raw_s + host, 3),
               "fold_ratio_raw": round(CELL_S / (raw_s + host), 4)}
        for lo_hi, f in (("lo", 0.63), ("hi", 0.73)):
            saved = (CELL_DEVICE_S - raw_s) * f
            out[f"fold_s_{lo_hi}"] = round(CELL_S - saved, 3)
            out[f"fold_ratio_{lo_hi}"] = round(CELL_S / (CELL_S - saved), 4)
        return out

    out = {
        "_what": "static per-op DRAM byte attribution of the tt-bio Boltz-2 pairformer block at "
                 "512 aa, c_z=128, counted off tt_bio/tenstorrent.py on the shipped path",
        "Z_bytes": Z, "N": N, "c_z": C_Z,
        "measured_block_Z": round(measured_Z, 1),
        "measured_block_GB": round(MEASURED_BLOCK_BYTES / 1e9, 3),
        "measured_block_ms": MEASURED_BLOCK_MS,
        "measured_block_GBps": round(MEASURED_BLOCK_BYTES / (MEASURED_BLOCK_MS / 1e3) / 1e9, 1),
        "measured_block_pct_of_roof": round(
            100 * MEASURED_BLOCK_BYTES / (MEASURED_BLOCK_MS / 1e3) / BW_ROOF, 1),
        "attributed_Z": round(attributed, 1),
        "attributed_pct_of_measured": round(100 * attributed / measured_Z, 1),
        "unattributed_Z": round(measured_Z - attributed, 1),
        "phases_Z": {
            "trimul_x2": round(trimul, 1),
            "trimul_input_side_x2": round(2 * tot(TRIMUL_IN), 1),
            "trimul_product_x2": round(2 * tot(TRIMUL_MID), 1),
            "trimul_tail_x2": round(2 * tot(TRIMUL_TAIL), 1),
            "triangle_attention_x2": round(triatt, 1),
            "transition_z": round(trans, 1),
            "residual_adds_x5": round(resid, 1),
            "s_track_pair_bias": round(strack, 1),
        },
        "bioir_same_phases_Z": {           # bioir_block_bytes.py, same counting rule
            "trimul_x2": 26, "triangle_attention_x2": 32,
            "transition_z": 10, "residual_adds_x5": 15, "bioir_block_Z": 83,
        },
        "per_op_trimul_start_Z": {n: r + w for n, r, w in TRIMUL_IN + TRIMUL_MID + TRIMUL_TAIL},
        "per_op_triatt_start_Z": {n: round(r + w, 3) for n, r, w in TRIMUL_IN[:0] + TRIATT
                                  if n not in ENDING_ONLY},
        "design": {
            "step1_mask_after_move_plus_E6": fold(step1),
            "step1_at_seq_pad_0_mask_dropped": fold(step1_nopad),
            "step1_2_L1_projection_into_gated_move": fold(step12),
            "step1_2_3_plus_F1_at_4_4": fold(step12 + step3),
            "step1_2_3_at_seq_pad_0": fold(step12_nopad + step3),
        },
    }
    p = Path(__file__).with_name("block_attrib.json")
    p.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
