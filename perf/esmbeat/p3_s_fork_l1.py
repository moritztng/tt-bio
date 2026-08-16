"""L1 pincer check for the fused pair-FFN kernel, against the tree's OWN budget model.

`_pair_proj_program_config` (tt_bio/tenstorrent.py) carries the CB budget every shipped pair-track
matmul is gated on, and the shipped L1_FC1 lever is the proof it is calibrated: that lever exists
precisely because the budget said out_block_w = 32 does not fit and 16 does. So the honest way to
screen a fused pair-FFN kernel is to add its CBs up in the same terms, not to re-estimate operand
sizes by hand.

No device needed. `_l1_bank_bytes()` reads 1461760 from the allocator on Blackhole, the source
quotes that constant, and the per-core bank does not change with the grid, so one number covers
qb1's 13x10 and qb2's 11x10 alike.

The penalty side is not modelled, it is read off `perf/esm3p4/screen_b_c2.json`: the whole FFN at
512 aa, rows = 32, measured on qb2 card 2 at every out_block_w the budget admits.
"""
import json

L1 = 1_461_760          # bytes per bank, the allocator number _l1_bank_bytes() returns
TILE = 1024 * 2         # bf16
FP32_PARTIAL = 4096     # the packer_l1_acc partial per tile, as the shipped formula counts it
SLACK = 128 * 1024      # the shipped formula's fixed term

# perf/esm3p4/screen_b_c2.json, qb2 card 2, 11x10, L = 512, n = 5, whole SwiGLUFFN, rows = 32.
# Every arm `torch.equal` True except obw = 32, which the allocator REFUSES outright.
FFN_MS_BY_OBW = {32: None, 16: 14.6623, 8: 17.1143, 4: 25.1120}


def matmul_cbs(bw, obh, obw, out_l1_tiles=0):
    """The shipped formula's terms for one matmul: in0/in1 double-buffered, the output block with
    its fp32 partial, the fixed slack, and an L1 destination if it has one."""
    return (2 * bw * (obh + obw) * TILE
            + obh * obw * (TILE + FP32_PARTIAL)
            + SLACK
            + out_l1_tiles * TILE)


def fused_ffn(bw, obh, obw, obw2):
    """One core's residency for a fused x_norm -> fc1a/fc1b -> SiLU*mul -> fc2 kernel.

    SwiGLU needs h1 and h2 live at the same time, so both fc1 halves carry an output block AND its
    fp32 partial concurrently. That is the term a hand estimate drops. fc2 contracts over d_ff, so
    it brings its own in1 and its own output block. The 128 KB slack is counted once, not per
    matmul: it is one kernel.

    The fp32 partial cannot be dropped by widening the contraction block. bw > 1 finishes the
    contraction in fewer rounds but is NOT bit-exact: all 20 bw = 1 arms of screen_a_c2.json are
    `torch.equal` against the shipped call and all 60 above it differ by one bf16 ulp.
    """
    in0 = 2 * bw * obh * TILE                       # x_norm block, shared by both fc1 halves
    w1 = 2 * (2 * bw * obw * TILE)                  # w1a and w1b in1 CBs
    h = 2 * (obh * obw * (TILE + FP32_PARTIAL))     # h1 and h2, live together for the gate
    w2 = 2 * bw * obw2 * TILE                       # fc2 in1
    out = obh * obw2 * (TILE + FP32_PARTIAL)        # fc2 output block
    total = in0 + w1 + h + w2 + out + SLACK
    d = {"in0": in0, "w1a+w1b": w1, "h1+h2": h, "w2": w2, "fc2_out": out,
         "slack": SLACK, "total": total, "fits": total <= L1,
         "pct_of_bank": round(100.0 * total / L1, 1)}
    ms = FFN_MS_BY_OBW.get(obw)
    if ms is not None and FFN_MS_BY_OBW[16] is not None:
        d["blocking_penalty_ms_per_call"] = round(ms - FFN_MS_BY_OBW[16], 4)
    return d


# Production shape at 512 aa, rows = 32: x [1,32,512,256] -> m_tiles 512, k_tiles 8, n_tiles 32
# (d_ff 1024). fc2 is [.., d_ff] x [d_ff, c_z] -> n_tiles 8. The shipped fc1 runs bw = 1,
# obw = 16, obh = 5, per_core_M = 5 on both grids: ceil(512/130) and ceil(512/110) both round to 5.
shipped_need = matmul_cbs(1, 5, 16, out_l1_tiles=5 * 32)

# The prize side, from state 8.6 re-priced for lever F: fc1 2.04 + fc2 0.55 + layer_norm 0.61 +
# slice 0.41 = 3.61 ms/call of headroom above the compute and SFPU floors, with the multiply's
# 1.35 ms above the SFPU floor deliberately excluded (1.3 measured there is no spare SFPU to hide
# the SiLU behind). This codebase's own fused kernels realise 50-80 % of such a ceiling.
CEILING_MS = 3.61
prize_lo, prize_hi = 0.50 * CEILING_MS, 0.80 * CEILING_MS

out = {
    "l1_bank_bytes": L1,
    "ffn_ms_by_obw_qb2": FFN_MS_BY_OBW,
    "shipped_fc1_l1_out": {
        "bw": 1, "obh": 5, "obw": 16, "per_core_M": 5, "n_tiles": 32,
        "need": shipped_need, "fits": shipped_need <= L1,
        "pct_of_bank": round(100.0 * shipped_need / L1, 1),
    },
    "fused_at_shipped_blocking": fused_ffn(bw=1, obh=5, obw=16, obw2=8),
    "fused_obw8": fused_ffn(bw=1, obh=5, obw=8, obw2=8),
    "fused_obw4": fused_ffn(bw=1, obh=5, obw=4, obw2=8),
    "verdict": {
        "ceiling_ms_per_call": CEILING_MS,
        "prize_ms_per_call": [round(prize_lo, 3), round(prize_hi, 3)],
        "widest_obw_that_fits": 8,
        "penalty_at_that_obw_ms_per_call": round(
            FFN_MS_BY_OBW[8] - FFN_MS_BY_OBW[16], 4),
        "net_ms_per_call": [
            round(prize_lo - (FFN_MS_BY_OBW[8] - FFN_MS_BY_OBW[16]), 3),
            round(prize_hi - (FFN_MS_BY_OBW[8] - FFN_MS_BY_OBW[16]), 3)],
        "net_s_per_fold_538_calls": [
            round(538 * (prize_lo - (FFN_MS_BY_OBW[8] - FFN_MS_BY_OBW[16])) / 1000, 3),
            round(538 * (prize_hi - (FFN_MS_BY_OBW[8] - FFN_MS_BY_OBW[16])) / 1000, 3)],
    },
}
print(json.dumps(out, indent=1))
