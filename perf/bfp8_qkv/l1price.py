#!/usr/bin/env python3
"""Price the `_qkv_l1_config` bfp8 headroom reclaim WITHOUT taking a card.

Replicates `_tri_att_qkv_l1_config`'s two fit tests from tt_bio/tenstorrent.py verbatim, then
asks the only question that matters: does budgeting in bfp8 tile bytes (1088) instead of the
hardcoded bf16 element width (2 -> tile 2048) move ANY shipped shape across the fit test onto
the L1-resident path, where the recorded win is 0.519 ms -> 0.264 ms?

Replication is checked against the live function separately; this file is the sweep.
"""
FP32_ACC_TILE = 4096
SLACK = 128 << 10
GRID = (11, 10)
NUM_CORES = GRID[0] * GRID[1]

# `_matmul_cb_budget()`: allocator number with a device, static number without. Both reported.
BUDGETS = {"allocator_1461760": 1461760, "unreserved_1532448": 1532448}


def cb_bytes(in0_block_w, out_block_h, out_block_w, tile, buffered=True, extra_tiles=0):
    return ((2 if buffered else 1) * in0_block_w * (out_block_h + out_block_w) * tile
            + out_block_h * out_block_w * (tile + FP32_ACC_TILE)
            + SLACK + extra_tiles * tile)


def fits(m_tiles, k_tiles, n_tiles, tile, l1):
    """Faithful port of _tri_att_qkv_l1_config's accept/reject, returning (ok, why, detail)."""
    if k_tiles >= NUM_CORES or m_tiles < n_tiles * 8 or n_tiles > 64:
        return False, "shape gate", {}
    per_core_M = next(
        (p for p in range(max(1, -(-m_tiles // NUM_CORES)), m_tiles + 1) if m_tiles % p == 0), 0)
    if not per_core_M or -(-m_tiles // per_core_M) > NUM_CORES:
        return False, "per_core_M", {}
    extra = -(-(m_tiles * n_tiles) // NUM_CORES)
    need = cb_bytes(k_tiles, per_core_M, n_tiles, tile, buffered=False, extra_tiles=extra)
    d = {"per_core_M": per_core_M, "extra_tiles": extra, "need": need, "l1": l1,
         "aggregate": 2 * m_tiles * n_tiles * tile, "agg_cap": 0.6 * NUM_CORES * l1}
    if need > l1:
        return False, "cb_bytes %d > %d (x%.2f over)" % (need, l1, need / l1), d
    if d["aggregate"] > d["agg_cap"]:
        return False, "aggregate L1 %d > %.0f" % (d["aggregate"], d["agg_cap"]), d
    return True, "FITS", d


GEOMS = [("docstring c_z=256 W=768", 256, 768), ("measured prod c_z=128 W=384", 128, 384)]
TOKENS = [32, 48, 64, 80, 96, 112, 128, 160, 192, 256, 384, 512]
TILES = [("bf16 (elem_bytes=2, tile 2048)", 2048), ("bfp8 (tile 1088)", 1088)]

for bname, l1 in BUDGETS.items():
    print("=" * 78)
    print("BUDGET %s = %d B/bank, grid %s, %d cores" % (bname, l1, GRID, NUM_CORES))
    for gname, cz, wqkv in GEOMS:
        print("\n  %s   k_tiles=%d n_tiles=%d" % (gname, cz // 32, wqkv // 32))
        print("    %-7s %-9s | %-44s | %s" % ("tokens", "m_tiles", TILES[0][0], TILES[1][0]))
        for tok in TOKENS:
            m = tok * tok
            if m % 32:
                continue
            row = []
            for _, tile in TILES:
                ok, why, d = fits(m // 32, cz // 32, wqkv // 32, tile, l1)
                row.append("%-6s %s" % ("FIT" if ok else "no", why))
            print("    %-7d %-9d | %-44s | %s" % (tok, m // 32, row[0], row[1]))
