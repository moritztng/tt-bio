#!/usr/bin/env python3
"""The two reblock sites' executed shapes, the fused-epilogue byte model, and the prediction.

CPU only, no ttnn, no device. Everything here is either (a) transcribed from `tt_bio/` with a
file:line citation, or (b) read out of `perf/c12_genop_rate/headroom.json`, which is the in-situ
per-site table measured inside the fold at a during-sampled 1350 MHz.

Why a transcription and not an import: `tt_bio.tenstorrent` needs ttnn, pc has no wheel installed,
and this row's first pass takes no card. The functions transcribed below are pure arithmetic over
(seq_len, hidden, batch, grid); each carries the line it came from so a drift fails review rather
than silently re-deriving a different plan. The 512 aa row is cross-checked against the executed
graph (`insitu_sites.py`'s 14-op-per-layer witness) at the bottom.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

# --- constants, tenstorrent.py -------------------------------------------------------------------
TRIANGLE_MULT_CHUNK_SIZE = 32                 # :22
SEQ_LEN_MORE_CHUNKING = 1536                  # :158
TRIMUL_IN_NORM_ROWBLOCK_BYTES = 3 * 2 ** 30   # :189
_TRIMUL_INPROJ_GROUP = 12                     # :354
_TRIMUL_INPROJ_FUSED_BYTES = 1024 * 2 ** 20   # :369
TRIANGLE_MULT_L1_CHUNK_BUDGET = 64 * 320 * 320  # :587
TRIANGLE_MULT_L1_MAX_SEQ = 352                # :580
COMPUTE_GRID_X_13 = 13                        # :908
GRID = (11, 10)                               # CORE_GRID_MAIN, :911 -- p300c's 110 cores
_MM_DEFAULT = (8, 8, 8, 2, 2)                 # :6851
TILE = 32
# reblock_permute.py
L1_N_MIN, L1_N_MAX = 288, 352                 # :338-339
GROUP_TILES = 32                              # :38
# Boltz-2 trimul: c_z = 128 contraction, hidden = 128 (g_in/p_in are [c_z, 2*hidden])
C_Z, HIDDEN = 128, 128


def trimul_chunk_size(seq_len, hidden, batch=1, grid=GRID):
    """tenstorrent.py:1050. Not --fast, not a small grid, so `_sq = seq_len`."""
    if seq_len > TRIANGLE_MULT_L1_MAX_SEQ:
        return TRIANGLE_MULT_CHUNK_SIZE
    gx, gy = grid
    budget = TRIANGLE_MULT_L1_CHUNK_BUDGET * gx * gy / (COMPUTE_GRID_X_13 * 10)
    c = TRIANGLE_MULT_CHUNK_SIZE
    while hidden % (c * 2) == 0 and batch * (c * 2) * seq_len * seq_len <= budget:
        c *= 2
    return c


def trimul_inproj_group(seq_len, chunk, batch, n_pairs):
    """tenstorrent.py:372. No refusal recorded, so the budget is the flat cap."""
    fused = 4 * chunk * seq_len * seq_len * batch * 2
    for g in range(min(n_pairs, _TRIMUL_INPROJ_GROUP), 1, -1):
        if n_pairs % g == 0 and g * fused <= _TRIMUL_INPROJ_FUSED_BYTES:
            return g
    return 1


def plan(seq_len, batch=1, grid=GRID):
    """Which branch the trimul channel loop takes at this padded token count."""
    H = seq_len
    large_seq = H > TRIANGLE_MULT_L1_MAX_SEQ           # _triangle_mul_memory_config, :1044
    row_norm = batch * H * H * C_Z * 2 > TRIMUL_IN_NORM_ROWBLOCK_BYTES
    chunk = trimul_chunk_size(H, HIDDEN, batch, grid)
    n_pairs = HIDDEN // chunk
    group = trimul_inproj_group(H, chunk, batch, n_pairs) if large_seq else 1
    slice_c = chunk * group
    iters = n_pairs // group
    # __call__, :6455 and :6529 -- the gated move needs a DRAM channel-loop memory config.
    gated = large_seq and not row_norm
    # _channel_move -> reblock_permute.eligible, :342
    fwd_ok = (H >= 256) if large_seq else (L1_N_MIN <= H <= L1_N_MAX)
    # _transform_chunk's `decompose`, :6108 -- only the DRAM path (or TAIL_L1, off by default)
    # splits (0,3,2,1) into the channel move plus a transpose, so on the L1 path the ENDING
    # variant's perm_a never reaches the kernel at all.
    fwd_calls_per_trimul = 2 if large_seq else 1
    # _channel_move_back -> reblock_permute.eligible_back, :579 -- DRAM only, N >= 256, N % 32 == 0
    back_ok = large_seq and H >= 256 and H % TILE == 0
    return dict(seq=seq_len, H=H, batch=batch, grid=list(grid), large_seq=large_seq,
                row_norm=row_norm, chunk=chunk, n_pairs=n_pairs, group=group,
                slice_c=slice_c, channel_loop_iters=iters,
                gp_width=4 * slice_c, gated_serves=gated,
                gated_calls_per_trimul=2 * iters if gated else 0,
                plain_fwd_serves=(not gated) and fwd_ok,
                plain_fwd_calls_per_trimul=fwd_calls_per_trimul * iters if
                ((not gated) and fwd_ok) else 0,
                back_serves=back_ok, back_calls_per_trimul=iters if back_ok else 0,
                branch="gated-move" if gated else ("four-way-split + plain move" if fwd_ok
                                                   else "four-way-split + ttnn.permute"))


def mm_cores(H, slice_c, grid=GRID):
    """mm_generic.build, :190-210, at the in-projection's shape. Z = one [1,C,H,H] bf16 tensor."""
    gx, gy = grid
    M_tiles = H * H // TILE
    K_tiles = C_Z // TILE
    N_tiles = 4 * slice_c // TILE
    M_block, K_block, N_block = _MM_DEFAULT[0], _MM_DEFAULT[1], _MM_DEFAULT[2]
    transpose = (H * H) > (4 * slice_c)
    in0_axis, in1_axis = (gx, gy) if transpose else (gy, gx)
    pad_M = -(-M_tiles // in0_axis) * in0_axis
    pad_N = -(-N_tiles // in1_axis) * in1_axis
    return dict(M_tiles=M_tiles, K_tiles=K_tiles, N_tiles=N_tiles, transpose=transpose,
                in0_axis_cores=in0_axis, in1_axis_cores=in1_axis,
                M_tiles_per_core=pad_M // in0_axis, N_tiles_per_core=pad_N // in1_axis,
                N_cores_with_real_output=-(-N_tiles // (pad_N // in1_axis)),
                K_blocks=-(-K_tiles // K_block) or 1,
                # the reblock writer's unit: 32 source m-tiles sharing one j-tile
                reblock_groups=M_tiles // GROUP_TILES,
                groups_per_M_core_max=-(-(M_tiles // GROUP_TILES) // in0_axis),
                groups_per_M_core_min=(M_tiles // GROUP_TILES) // in0_axis)


def bytes_model(H, slice_c):
    """DRAM bytes per trimul call, in Z = H*H*slice_c*2 units, from the kernels' own addressing.

    today  in-proj  read in0 once (one sender core per M index, multicast down the column)  1 Z
                    write the fused projection [1,H,H,4*slice_c]                            4 Z
           gated x2 read the p slice and the g slice, write one [1,slice_c,H,H]         2 x 3 Z
    fused  read in0 once, write `a` and `b`                                                 3 Z
    """
    Z = H * H * slice_c * 2
    return dict(Z_bytes=Z, today_inproj_Z=5, today_gated_Z=6, today_Z=11, fused_Z=3,
                deleted_Z=8, today_MB=11 * Z / 1e6, fused_MB=3 * Z / 1e6)


def main():
    head = json.load(open(HERE.parent / "c12_genop_rate" / "headroom.json"))
    site = {s["site"]: s for s in head["sites"]}
    bw_1r1w = head["roofs"]["bw_1r1w"]

    sizes = {}
    for seq in (298, 512, 768):
        H = -(-seq // TILE) * TILE          # the token axis buckets to a multiple of 32
        p = plan(H)
        p["seq_aa"] = seq
        p["mm"] = mm_cores(H, p["slice_c"])
        p["bytes"] = bytes_model(H, p["slice_c"])
        sizes[seq] = p

    # --- the prediction, at 512 aa, pre-registered -----------------------------------------------
    g, i = site["reblock_gated"], site["trimul_in"]
    calls_gated, calls_in = g["calls"], i["calls"]
    today_ms = i["ms"] + 2 * g["ms"]                       # per trimul call
    today_s = i["sec"] + g["sec"]
    Zb = sizes[512]["bytes"]["Z_bytes"]
    fused_floor_ms = 3 * Zb / (bw_1r1w * 1e6)              # 1 read + 2 writes
    eff_gated = g["floor_ms"] / g["ms"]                    # 0.794, the better parent
    eff_in = i["floor_ms"] / i["ms"]                       # 0.619, the worse parent
    arms = {
        "optimistic": fused_floor_ms / eff_gated,
        "central": fused_floor_ms / eff_in,
        # gate SFPU stops hiding: traffic floor + the matmul's own arithmetic, then the
        # reblock writer's measured 20.6 % exposure on top
        "pessimistic": (fused_floor_ms + i["arith_ms"]) / eff_gated,
    }
    # --- arm 1a: the gate epilogue WITHOUT the reblock in the writer --------------------------
    # The matmul writes p * sigmoid(g) into `a` and `b` in the ORIGINAL [1,H,H,slice_c] layout, so
    # its writer keeps its normal tile order and only the destination addressing changes. The two
    # plain forward moves then run as today's `reblock_permute`, which is arithmetic-free and
    # bit-exact by construction. 3 Z + 2 x 2 Z = 7 Z against today's 11 Z.
    # The plain move's cost is ESTIMATED from `reblock_back`, which moves the same 2 Z between the
    # same two shapes in the inverse direction and is measured at 0.4941 ms/call. It is not a
    # measurement of the forward move at this shape.
    back = site["reblock_back"]
    Z = Zb / 1e6
    move_floor_ms = 2 * Zb / (bw_1r1w * 1e6)
    a1 = {
        "optimistic": arms["optimistic"] + 2 * move_floor_ms / eff_gated,
        "central": arms["central"] + 2 * back["ms"],
        "pessimistic": arms["pessimistic"] + 2 * back["ms"],
    }

    pred = {}
    for k, ms in arms.items():
        sec = ms * calls_in / 1000.0
        pred[k] = dict(ms_per_call=ms, fold_s=sec, fold_Mc=sec * 1350.0,
                       saved_s=today_s - sec, saved_Mc=(today_s - sec) * 1350.0)

    pred_1a = {}
    for k, ms in a1.items():
        sec = ms * calls_in / 1000.0
        pred_1a[k] = dict(ms_per_call=ms, fold_s=sec, fold_Mc=sec * 1350.0,
                          saved_s=today_s - sec, saved_Mc=(today_s - sec) * 1350.0)

    out = dict(
        clock_mhz=1350, grid=list(GRID), part="p300c Blackhole",
        source="perf/c12_genop_rate/headroom.json @ 4ed84475d (in situ, during-sampled)",
        sizes=sizes,
        site_512={
            "reblock_gated": dict(calls=calls_gated, ms=g["ms"], sec=g["sec"],
                                  Mc=g["sec"] * 1350, floor_ms=g["floor_ms"],
                                  pct_roof=g["pct_roof"]),
            "reblock_back": dict(calls=site["reblock_back"]["calls"],
                                 ms=site["reblock_back"]["ms"],
                                 sec=site["reblock_back"]["sec"],
                                 Mc=site["reblock_back"]["sec"] * 1350,
                                 floor_ms=site["reblock_back"]["floor_ms"],
                                 pct_roof=site["reblock_back"]["pct_roof"]),
            "trimul_in": dict(calls=calls_in, ms=i["ms"], sec=i["sec"], Mc=i["sec"] * 1350,
                              arith_ms=i["arith_ms"], floor_ms=i["floor_ms"]),
        },
        fused=dict(replaces=["trimul_in", "reblock_gated"], today_ms_per_call=today_ms,
                   today_fold_s=today_s, today_fold_Mc=today_s * 1350,
                   floor_ms_per_call=fused_floor_ms, bw_1r1w_GBs=bw_1r1w,
                   eff_gated=eff_gated, eff_inproj=eff_in, predicted=pred),
        arm_1a=dict(what="gate epilogue only, plain reblock_permute keeps the move",
                    today_Z=11, arm_Z=7, deleted_Z=4,
                    move_ms_estimated_from="reblock_back, same 2 Z, inverse direction",
                    move_ms=back["ms"], predicted=pred_1a),
        untouched=dict(reblock_back_s=site["reblock_back"]["sec"],
                       reblock_back_Mc=site["reblock_back"]["sec"] * 1350,
                       why="its producer is ttnn.matmul (tenstorrent.py:6573), not a tt-bio "
                           "generic_op, so there is no wheel route to its writer"),
    )
    json.dump(out, open(HERE / "sites.json", "w"), indent=1)

    print(f"grid {GRID}  clock 1350 MHz  part p300c Blackhole\n")
    print("per-size plan (H = padded token count)")
    print("  aa    H  path  chunk group slice_c  iters  gp_w  gated/tri  plainfwd/tri  back/tri")
    for seq, p in sizes.items():
        print("  %3d %4d  %-4s %5d %5d %7d %6d %5d %10d %13d %9d   %s"
              % (seq, p["H"], "DRAM" if p["large_seq"] else "L1", p["chunk"], p["group"],
                 p["slice_c"], p["channel_loop_iters"], p["gp_width"],
                 p["gated_calls_per_trimul"], p["plain_fwd_calls_per_trimul"],
                 p["back_calls_per_trimul"], p["branch"]))
    m = sizes[512]["mm"]
    print("\nin-projection core plan at 512 aa: M %d tiles over %d cores = %d each, "
          "N %d tiles over %d cores = %d each, %d of %d N-cores carry real output"
          % (m["M_tiles"], m["in0_axis_cores"], m["M_tiles_per_core"], m["N_tiles"],
             m["in1_axis_cores"], m["N_tiles_per_core"], m["N_cores_with_real_output"],
             m["in1_axis_cores"]))
    print("  reblock writer groups (32 m-tiles, one j-tile): %d over %d M-cores = %d..%d each, "
          "imbalance %.1f %%"
          % (m["reblock_groups"], m["in0_axis_cores"], m["groups_per_M_core_min"],
             m["groups_per_M_core_max"],
             100 * (m["groups_per_M_core_max"] / (m["reblock_groups"] / m["in0_axis_cores"]) - 1)))
    b = sizes[512]["bytes"]
    print("\nbytes per trimul call at 512 aa, Z = %.4f MB: today %d Z = %.1f MB, "
          "fused %d Z = %.1f MB, deleted %d Z"
          % (b["Z_bytes"] / 1e6, b["today_Z"], b["today_MB"], b["fused_Z"], b["fused_MB"],
             b["deleted_Z"]))
    print("\ntoday (in situ, 1350 MHz): trimul_in %.4f s + reblock_gated %.4f s = %.4f s "
          "(%.1f Mc) over %d + %d calls" % (i["sec"], g["sec"], today_s, today_s * 1350,
                                            calls_in, calls_gated))
    print("fused traffic floor %.4f ms/call at %.1f GB/s; parents run at %.1f %% and %.1f %% "
          "of their own floors" % (fused_floor_ms, bw_1r1w, 100 * eff_gated, 100 * eff_in))
    print("\nPREDICTED fold seconds for the fused op, 560 calls:")
    for k in ("optimistic", "central", "pessimistic"):
        q = pred[k]
        print("  %-12s %.4f ms/call -> %.4f s (%.1f Mc), saves %.4f s (%.1f Mc)"
              % (k, q["ms_per_call"], q["fold_s"], q["fold_Mc"], q["saved_s"], q["saved_Mc"]))
    print("\nARM 1a, gate epilogue only (7 Z, plain move kept, move cost estimated from "
          "reblock_back):")
    for k in ("optimistic", "central", "pessimistic"):
        q = pred_1a[k]
        print("  %-12s %.4f ms/call -> %.4f s (%.1f Mc), saves %.4f s (%.1f Mc)"
              % (k, q["ms_per_call"], q["fold_s"], q["fold_Mc"], q["saved_s"], q["saved_Mc"]))
    print("\nreblock_back stays: %.4f s (%.1f Mc). %s"
          % (out["untouched"]["reblock_back_s"], out["untouched"]["reblock_back_Mc"],
             out["untouched"]["why"]))
    # executed-graph cross-check: 14 generic_ops per layer, 264 pfl + 16 msal legs
    assert calls_in == 560 and calls_gated == 1120 and site["reblock_back"]["calls"] == 560
    assert sizes[512]["gated_calls_per_trimul"] == 2, sizes[512]
    assert sizes[512]["back_calls_per_trimul"] == 1, sizes[512]
    assert sizes[298]["gated_calls_per_trimul"] == 0, sizes[298]
    print("\nexecuted-graph cross-check: 2 gated + 1 back per trimul at 512 aa, and 560 trimul "
          "calls (280 layers x 2) reproduces 1120 / 560. OK")


if __name__ == "__main__":
    main()
