// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute GATHER writer. Produces output = permute(x, (0,3,1,2)) for
// x [1, N, N, C] -> [1, C, N, N], C a multiple of 32, Ct = C/32 channel tiles.
//
// The compute kernel feeds, per group (it, jt) and channel tile ct, the 32
// post-WH tiles in il-ascending order, where tile `il` is
// WHtile_il[c, kl] = x[it*32+il, jt*32+kl, ct*32+c] (row = channel-within-tile c,
// col = j-within-tile kl).
//
// Output tile (ch_plane = ct*32+c, it, jt) has within-tile element
//   O_ch[row=il, col=kl] = x[it*32+il, jt*32+kl, ch] = WHtile_il[c, kl].
// So O_ch row `il` is exactly row `c` of WHtile_il. There is NO within-tile
// transpose — just gather row `c` from each of the 32 source tiles and stack them
// as the 32 rows of O_ch.
//
// KEY (Blackhole): a 2-byte strided noc_async_write to DRAM is NOT 16B-aligned
// and gets quantized to 16B blocks (scrambles data). FIX = assemble the whole
// output tile in an L1 staging CB via local (alignment-free) L1 copies, then do
// ONE aligned, contiguous 2KB tile write to DRAM. We copy FACE_WIDTH(16) elems
// at a time (one face-row), which is the contiguous unit of standard tile layout.
//
// Double-buffered staging (c_24 depth 2) lets the DRAM write of channel c
// overlap the L1 gather of channel c+1. noc_async_writes_flushed() before
// reusing a staging slot guarantees the prior write drained off that slot.
//
// Back-ported from the v0.74 object dataflow API (Noc / CircularBuffer / CoreLocalMem) to the
// free-function API the production ttnn wheels ship, so this kernel JIT-compiles under
// ttnn.generic_op with no tt-metal source build (verified on 0.68.0 and on 0.67.4). The
// transaction structure is unchanged.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

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
    uint32_t Ct = get_arg_val<uint32_t>(4);

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

    const auto s = TensorAccessor(dst_args, dst_addr);

    // Manually ping-pong between the two tiles of the depth-2 staging CB. We
    // reserve the whole CB once and treat it as fixed scratch (no per-iter
    // CB push/pop): slot s alternates 0/1 so the DRAM write of channel c
    // overlaps the L1 gather of channel c+1. noc_async_writes_flushed() before
    // reusing a slot guarantees the prior write to that slot has drained.
    cb_reserve_back(stage_cb_id, 2);
    const uint32_t stage_base0 = get_write_ptr(stage_cb_id);

    // Track whether each slot has an outstanding write that must be flushed
    // before reuse. Slots are first used (no prior write) so start "clean".
    bool slot_dirty[2] = {false, false};

    const uint32_t NtNt = Nt * Nt;
    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t it = group / Nt;
        const uint32_t jt = group % Nt;
        const uint32_t page_base = it * Nt + jt;

        // Rows [rows_valid, 32) of every output tile in this group are tile padding. They must be
        // zero (this tensor is a matmul operand and the padding sits on the contracted axis), they
        // are the same rows for every channel of every channel tile, and the DRAM write never
        // modifies the slot -- so zero both staging slots once per group and leave them alone.
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
                        for (uint32_t k = 0; k < (FACE_WIDTH * element_size) / 4; ++k) {
                            z[k] = 0;
                        }
                    }
                }
            }
        }

        for (uint32_t ct = 0; ct < Ct; ++ct) {
            // Wait for all 32 post-WH tiles of this (group, ct) (resident contiguously in c_16).
            cb_wait_front(cb_id_in, TILE_HEIGHT);
            const uint32_t group_l1_base = get_read_ptr(cb_id_in);

            for (uint32_t c = 0; c < TILE_HEIGHT; ++c) {
                const uint32_t ch_face_h = c / FACE_HEIGHT;   // which face-row the channel sits in
                const uint32_t ch_in_face = c % FACE_HEIGHT;  // channel offset within that face

                const uint32_t slot = c & 1u;
                const uint32_t stage_base = stage_base0 + slot * tile_bytes;
                // If this slot still has an outstanding DRAM write, drain it first.
                if (slot_dirty[slot]) {
                    noc_async_writes_flushed();
                }
                // Assemble O_ch: row il = row c of WHtile_il. Each face-row is
                // FACE_WIDTH contiguous bf16 = 32 bytes, 16B-aligned. Issue the 64
                // (= 32 il x 2 face_w) local L1->L1 copies as ASYNC NoC reads (read
                // datamover) so the NoC pipelines them instead of serializing scalar
                // stores on the RISC; one barrier drains them before the DRAM write.
                constexpr uint32_t FACE_ROW_BYTES = FACE_WIDTH * element_size;  // 32
                for (uint32_t il = 0; il < rows_valid; ++il) {
                    const uint32_t src_tile_base = group_l1_base + il * tile_bytes;
                    const uint32_t il_face_h = il / FACE_HEIGHT;   // dest face-row
                    const uint32_t il_in_face = il % FACE_HEIGHT;  // dest offset within face
                    for (uint32_t face_w = 0; face_w < NUM_FACES_W; ++face_w) {
                        const uint32_t src_elem = (ch_face_h * NUM_FACES_W + face_w) *
                                                      face_height_width + ch_in_face * FACE_WIDTH;
                        const uint32_t dst_elem = (il_face_h * NUM_FACES_W + face_w) *
                                                      face_height_width + il_in_face * FACE_WIDTH;
                        const uint32_t src_l1 = src_tile_base + src_elem * element_size;
                        const uint32_t dst_l1 = stage_base + dst_elem * element_size;
                        // 16B-aligned (offsets are multiples of FACE_WIDTH=16 elems=32B),
                        // async, read-datamover, max xfer 32B.
                        tt::data_movement::common::tt_memmove<true, true, true, FACE_ROW_BYTES>(
                            dst_l1, src_l1, FACE_ROW_BYTES);
                    }
                }
                noc_async_read_barrier();  // drain the L1->L1 gather before the DRAM write

                // One aligned, contiguous 2KB tile write to DRAM page
                //   page = (ct*32 + c) * Nt*Nt + it*Nt + jt.
                const uint32_t out_page = (ct * TILE_HEIGHT + c) * NtNt + page_base;
                noc_async_write(stage_base, s.get_noc_addr(out_page), tile_bytes);
                slot_dirty[slot] = true;
            }

            // Drain all writes of this channel tile before popping its source tiles.
            noc_async_write_barrier();
            slot_dirty[0] = false;
            slot_dirty[1] = false;
            cb_pop_front(cb_id_in, TILE_HEIGHT);
        }
    }
}
