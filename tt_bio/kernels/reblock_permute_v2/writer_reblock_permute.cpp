// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute GATHER writer. Produces output = permute(x, (0,3,1,2)) for
// x [1, N, N, 32] -> [1, 32, N, N].
//
// The compute kernel feeds, per group (it, jt), the 32 post-WH tiles in
// il-ascending order, where tile `il` is WHtile_il[ch, kl] = x[it*32+il,
// jt*32+kl, ch] (row = channel ch, col = j-within-tile kl).
//
// Output tile (ch_plane=ch, it, jt) has within-tile element
//   O_ch[row=il, col=kl] = O[ch, it*32+il, jt*32+kl] = x[it*32+il, jt*32+kl, ch]
//                        = WHtile_il[ch, kl].
// So O_ch row `il` is exactly row `ch` of WHtile_il. There is NO within-tile
// transpose (rows il<->kl identity on columns) — just gather row `ch` from each
// of the 32 source tiles and stack them as the 32 rows of O_ch.
//
// KEY (Blackhole): a 2-byte strided noc_async_write to DRAM is NOT 16B-aligned
// and gets quantized to 16B blocks (scrambles data). FIX = assemble the whole
// output tile in an L1 staging CB via local (alignment-free) L1 copies, then do
// ONE aligned, contiguous 2KB tile write to DRAM. We copy FACE_WIDTH(16) elems
// at a time (one face-row), which is the contiguous unit of standard tile layout.
//
// Double-buffered staging (c_24 depth 2) lets the DRAM write of channel ch
// overlap the L1 gather of channel ch+1. noc_async_writes_flushed() before
// reusing a staging slot guarantees the prior write drained off that slot.
//
// THE BINDING RESOURCE IS INSTRUCTION COUNT ON THIS RISC, not bytes: 64 NOC
// transactions per source tile is a floor over all kernel structures (proved by
// P4's bf16 -> fp32 test, which left the time unchanged while the byte-bound
// control rose 1.36x), so the only axis left is instructions per transaction.
// Two things are done for it here and neither changes the transaction count:
//
//   * the gather loop issues `noc_async_read_one_packet_with_state`, with the NOC
//     coordinates and the 32-byte length written ONCE before the group loop.
//     Every read in this kernel is local L1 -> L1 at a fixed size, so the only
//     per-transaction state is the two local addresses.
//   * every address in the gather loop is an induction variable. The source
//     offset is invariant in `il`, the destination advances one face-row per
//     `il`, and the loop is split at the face-row boundary so the jump between
//     face rows is not a branch inside the loop. No multiply, no divide and no
//     conditional survives in the body.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // See the reader: the destination address is the only per-call value, so it is a common
    // runtime arg and the descriptor above it is cacheable.
    uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t start_group = get_arg_val<uint32_t>(0);
    uint32_t num_groups = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);
    // Logical length of the permuted axis; see the reader. Rows at or above it are output tile
    // padding and must be ZERO, not a copy of row 0: this tensor is a matmul operand and the
    // padding sits on the contracted axis, so a non-zero there changes the product.
    uint32_t D1 = get_arg_val<uint32_t>(3);

    constexpr uint32_t element_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(1);     // c_16 (post-WH tiles)
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(2);  // 32
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(3);   // 32
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(4);  // 16
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(5);   // 16
    constexpr uint32_t stage_cb_id = get_compile_time_arg_val(6);  // c_24
    constexpr auto dst_args = TensorAccessorArgs<7>();

    constexpr uint32_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;                 // 2
    constexpr uint32_t face_height_width = FACE_HEIGHT * FACE_WIDTH;          // 256
    constexpr uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;  // 2048
    constexpr uint32_t FACE_ROW_BYTES = FACE_WIDTH * element_size;            // 32
    constexpr uint32_t FACE_BYTES = face_height_width * element_size;         // 512

    const auto s = TensorAccessor(dst_args, dst_addr);

    // Manually ping-pong between the two tiles of the depth-2 staging CB. We
    // reserve the whole CB once and treat it as fixed scratch (no per-iter
    // CB push/pop): slot s alternates 0/1 so the DRAM write of channel ch
    // overlaps the L1 gather of channel ch+1. noc_async_writes_flushed() before
    // reusing a slot guarantees the prior write to that slot has drained.
    cb_reserve_back(stage_cb_id, 2);
    const uint32_t stage_base0 = get_write_ptr(stage_cb_id);

    // Every gather read is local L1 -> L1 at exactly FACE_ROW_BYTES, so the NOC coordinates and
    // the transfer length are the same for all of them. Write them into the read command buffer
    // once here; the loop below then writes only the two local addresses per transaction.
    noc_async_read_one_packet_set_state(get_noc_addr(stage_base0), FACE_ROW_BYTES);

    // Track whether each slot has an outstanding write that must be flushed
    // before reuse. Slots are first used (no prior write) so start "clean".
    bool slot_dirty[2] = {false, false};

    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t it = group / Nt;
        const uint32_t jt = group % Nt;

        // Wait for all 32 post-WH tiles of this group (resident contiguously in c_16).
        cb_wait_front(cb_id_in, TILE_HEIGHT);
        const uint32_t group_l1_base = get_read_ptr(cb_id_in);

        // Rows [rows_valid, 32) of every output tile in this group are tile padding. They must be
        // zero (this tensor is a matmul operand and the padding sits on the contracted axis), they
        // are the same rows for all 32 channels, and the DRAM write never modifies the slot -- so
        // zero both staging slots once, here, and leave them alone.
        const uint32_t row_base_g = it * TILE_HEIGHT;
        const uint32_t rows_valid = (row_base_g + TILE_HEIGHT <= D1) ? TILE_HEIGHT
                                                                     : (D1 - row_base_g);
        if (rows_valid < TILE_HEIGHT) {
            for (uint32_t slot = 0; slot < 2; ++slot) {
                const uint32_t sb = stage_base0 + slot * tile_bytes;
                for (uint32_t il = rows_valid; il < TILE_HEIGHT; ++il) {
                    const uint32_t il_face_h = il / FACE_HEIGHT;
                    const uint32_t il_in_face = il % FACE_HEIGHT;
                    for (uint32_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                        const uint32_t dst_elem = (il_face_h * NUM_FACES_W + face_w) *
                                                      face_height_width + il_in_face * FACE_WIDTH;
                        volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                            sb + dst_elem * element_size);
                        for (uint32_t k = 0; k < FACE_ROW_BYTES / 4; ++k) {
                            z[k] = 0;
                        }
                    }
                }
            }
        }

        // Rows of the destination split at the face-row boundary: rows [0,16) live in dest faces
        // 0 and 1, rows [16,32) in faces 2 and 3. Splitting the loop there keeps the jump between
        // them out of the body.
        const uint32_t rows_lo = rows_valid < FACE_HEIGHT ? rows_valid : FACE_HEIGHT;
        const uint32_t rows_hi = rows_valid > FACE_HEIGHT ? rows_valid - FACE_HEIGHT : 0;

        for (uint32_t ch = 0; ch < TILE_HEIGHT; ++ch) {
            const uint32_t slot = ch & 1u;
            const uint32_t stage_base = stage_base0 + slot * tile_bytes;
            // If this slot still has an outstanding DRAM write, drain it first.
            if (slot_dirty[slot]) {
                noc_async_writes_flushed();
            }

            // Source offset of channel `ch`, invariant in `il`: face-row (ch / 16) selects the
            // source face pair, (ch % 16) the row inside it.
            const uint32_t src_ch = group_l1_base
                                  + (ch / FACE_HEIGHT) * NUM_FACES_W * FACE_BYTES
                                  + (ch % FACE_HEIGHT) * FACE_ROW_BYTES;
            uint32_t s0 = src_ch;                // face_w = 0
            uint32_t s1 = src_ch + FACE_BYTES;   // face_w = 1
            uint32_t d0 = stage_base;
            uint32_t d1 = stage_base + FACE_BYTES;
            for (uint32_t il = 0; il < rows_lo; ++il) {
                noc_async_read_one_packet_with_state(s0, d0);
                noc_async_read_one_packet_with_state(s1, d1);
                s0 += tile_bytes;
                s1 += tile_bytes;
                d0 += FACE_ROW_BYTES;
                d1 += FACE_ROW_BYTES;
            }
            if (rows_hi) {
                s0 = src_ch + FACE_HEIGHT * tile_bytes;
                s1 = s0 + FACE_BYTES;
                d0 = stage_base + 2 * FACE_BYTES;
                d1 = stage_base + 3 * FACE_BYTES;
                for (uint32_t il = 0; il < rows_hi; ++il) {
                    noc_async_read_one_packet_with_state(s0, d0);
                    noc_async_read_one_packet_with_state(s1, d1);
                    s0 += tile_bytes;
                    s1 += tile_bytes;
                    d0 += FACE_ROW_BYTES;
                    d1 += FACE_ROW_BYTES;
                }
            }
            noc_async_read_barrier();  // drain the L1->L1 gather before the DRAM write

            // One aligned, contiguous 2KB tile write to DRAM page
            //   page = ch * (Nt*Nt) + it * Nt + jt.
            const uint32_t out_page = ch * (Nt * Nt) + it * Nt + jt;
            noc_async_write(stage_base, s.get_noc_addr(out_page), tile_bytes);
            slot_dirty[slot] = true;
        }

        // Drain all writes of this group before popping the source tiles.
        noc_async_write_barrier();
        slot_dirty[0] = false;
        slot_dirty[1] = false;
        cb_pop_front(cb_id_in, TILE_HEIGHT);
    }
}
