#!/usr/bin/env python3
"""K2: hold the triangle bias in a permanently fronted CB instead of re-reading it per batch.

The mask tile ids depend only on (head, q_chunk, k_chunk) -- `mask_batch_offset` is 0 whenever the
mask has a batch dimension of 1, which triangle attention's [1, H, S, S] bias does -- so the reader's
`nb` loop re-reads the same 4.19 MB 512 times, 2048 MiB/call against the 4 MiB the maths needs.

Two guarded edits, both against copies in tt_bio/kernels/triatt_sdpa/:

  reader_interleaved.cpp   fill every (q chunk, k chunk) block ONCE before the `nb` loop,
                           q-chunk-major then k-chunk-major, so block (lq, kc) stays contiguous at
                           tile offset (lq * PERSISTENT_MASK + kc) * Sq_chunk_t * Sk_chunk_t, and
                           skip the per-chunk fill inside the loop.
  compute_common.hpp       `add_block_inplace` gains an `in1_base` offset; the mask call site uses
                           `<false>` so it never pops, with the base derived from the core-local q
                           index. The function was already templated on `pop_in1`, so this is two
                           expressions and one default argument.

Guarded on PERSISTENT_MASK, whose value is k_num_chunks. Undefined, both files are the wheel's.

The host is responsible for the preconditions the fill assumes and asserts them before setting the
define: one head per core, a batch-broadcast mask, and no padded mask. It also sizes cb_mask_in to
`q_per_core * PERSISTENT_MASK * Sq_chunk_t * Sk_chunk_t` tiles, which is what lets a core own more
than one q chunk.
"""
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
reader = ROOT / "dataflow/reader_interleaved.cpp"
common = ROOT / "compute/compute_common.hpp"

# ---- compute -------------------------------------------------------------------------------------
s = common.read_text()
if "in1_base" in s:
    print("compute already patched")
else:
    old_sig = "void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles) {"
    new_sig = ("void add_block_inplace(uint32_t in0_cb, uint32_t in1_cb, uint32_t num_tiles,\n"
               "                       uint32_t in1_base = 0) {")
    assert s.count(old_sig) == 1, "add_block_inplace signature not found"
    s = s.replace(old_sig, new_sig, 1)

    old_wait = "    cb_wait_front(in1_cb, num_tiles);\n    for (uint32_t i = 0; i < num_tiles; i++) {\n        acquire_dst();\n        add_tiles(in0_cb, in1_cb, i, i, 0);"
    new_wait = "    cb_wait_front(in1_cb, in1_base + num_tiles);\n    for (uint32_t i = 0; i < num_tiles; i++) {\n        acquire_dst();\n        add_tiles(in0_cb, in1_cb, i, in1_base + i, 0);"
    assert s.count(old_wait) == 1, "add_block_inplace body not found"
    s = s.replace(old_wait, new_wait, 1)

    old_call = "                } else {\n                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);\n                }"
    new_call = ("                } else {\n"
                "#ifdef PERSISTENT_MASK\n"
                "                    // The core's whole mask is fronted once; index block\n"
                "                    // (local q chunk, k_chunk) and never pop, so the next batch reuses it.\n"
                "                    add_block_inplace<false>(\n"
                "                        cb_qk_im, cb_mask_in, qk_chunk_tiles, (pm_q_block + k_chunk) * qk_chunk_tiles);\n"
                "#else\n"
                "                    add_block_inplace(cb_qk_im, cb_mask_in, qk_chunk_tiles);\n"
                "#endif\n"
                "                }")
    assert s.count(old_call) == 1, "mask add call site not found"
    s = s.replace(old_call, new_call, 1)

    old_loop = ("    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {\n"
                "        uint32_t q_low_idx;\n"
                "        uint32_t q_high_idx;")
    new_loop = ("    for (uint32_t q_iter = iter_q_start; q_iter < iter_q_end; ++q_iter) {\n"
                "#ifdef PERSISTENT_MASK\n"
                "        // Base of this q chunk's fronted mask blocks. Both call sites define q_chunk as\n"
                "        // `local_q_start + (q_iter - iter_q_start)`, so this is the core-local q index times the\n"
                "        // k-chunk count. BALANCED_Q_PARALLEL would break that identity; the host never sets it.\n"
                "        const uint32_t pm_q_block = (q_iter - iter_q_start) * PERSISTENT_MASK;\n"
                "#endif\n"
                "        uint32_t q_low_idx;\n"
                "        uint32_t q_high_idx;")
    assert s.count(old_loop) == 1, "q_iter loop head not found"
    s = s.replace(old_loop, new_loop, 1)
    common.write_text(s)
    print("patched compute/compute_common.hpp")

# ---- reader --------------------------------------------------------------------------------------
r = reader.read_text()
if "PERSISTENT_MASK" in r:
    print("reader already patched")
    raise SystemExit(0)

FILL = '''
#ifdef PERSISTENT_MASK
    // K2: the mask depends only on (head, q_chunk, k_chunk), and this core owns one head and the
    // q chunks [local_q_start, local_q_end), so read every block it will ever need once here and
    // never refill. Blocks are q-chunk-major then k-chunk-major: block (lq, kc) starts at tile
    // ((lq - local_q_start) * PERSISTENT_MASK + kc) * mask_chunk_tiles, which is the base the
    // compute side indexes with. The host asserts the preconditions and sizes the CB to
    // q_chunks_per_core * PERSISTENT_MASK blocks: one head per core, a batch-broadcast mask, and
    // no padded mask.
    {
        const uint32_t persistent_mask_tiles = q_chunks_per_core * PERSISTENT_MASK * mask_chunk_tiles;
        cb_reserve_back(cb_mask_in, persistent_mask_tiles);
        uint32_t pm_write_ptr = get_write_ptr(cb_mask_in);
        uint32_t pm_barrier = 0;
        for (uint32_t lq = local_q_start; lq < local_q_end; ++lq) {
            for (uint32_t kc = 0; kc < PERSISTENT_MASK; ++kc) {
                uint32_t pm_row_start = lq * Sq_chunk_t * valid_Skt;
                if constexpr (!broadcast_provided_mask_heads) {
                    pm_row_start += local_nh_start * valid_Sqt * valid_Skt;
                }
                for (uint32_t row = 0; row < Sq_chunk_t; ++row) {
                    for (uint32_t col = 0; col < Sk_chunk_t; ++col) {
                        noc_async_read_tile(pm_row_start + kc * Sk_chunk_t + col, mask_reader, pm_write_ptr);
                        pm_write_ptr += mask_tile_bytes;
                        if (++pm_barrier == barrier_threshold) {
                            noc_async_read_barrier();
                            pm_barrier = 0;
                        }
                    }
                    pm_row_start += valid_Skt;
                }
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_mask_in, persistent_mask_tiles);
    }
#endif

'''

anchor = "        for (uint32_t nb = local_batch_start; nb < local_batch_end; ++nb) {"
assert r.count(anchor) == 1, "nb loop not found"
r = r.replace(anchor, FILL.lstrip("\n") + anchor, 1)

old_fill = """                        // Mask read — safe after linked write pair is complete
                        if constexpr (use_provided_mask) {"""
new_fill = """                        // Mask read — safe after linked write pair is complete
                        // PERSISTENT_MASK hoists this whole block out of the nb loop, above.
#ifdef PERSISTENT_MASK
                        if constexpr (false) {
#else
                        if constexpr (use_provided_mask) {
#endif"""
assert r.count(old_fill) == 1, "per-chunk mask fill not found"
r = r.replace(old_fill, new_fill, 1)

reader.write_text(r)
print("patched dataflow/reader_interleaved.cpp")
