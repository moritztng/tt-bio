// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// inproj_gated writer: writer_reblock_permute_gated.cpp's 32-row reblock, looped over the CPG
// channel tiles a group carries. A group is (it, jt, sub); channel tile ct = sub * CPG + q. The
// compute kernel pushes 32 c x j tiles per channel tile, one per position row, and each output
// tile (c, it, jt) of [1, C, N, N] gathers face row c of all 32 of them. See the gated writer for
// the staging and the padding rows; both are unchanged.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_common_arg_val<uint32_t>(0);
    uint32_t first_group = get_arg_val<uint32_t>(0);
    uint32_t num_groups = get_arg_val<uint32_t>(1);
    uint32_t Nt = get_arg_val<uint32_t>(2);
    uint32_t D1 = get_arg_val<uint32_t>(3);
    uint32_t Ct = get_arg_val<uint32_t>(4);
    uint32_t S = get_arg_val<uint32_t>(5);
    uint32_t group_stride = get_arg_val<uint32_t>(6);
    uint32_t group_wrap_hi = get_arg_val<uint32_t>(7);
    uint32_t group_wrap_lo = get_arg_val<uint32_t>(8);

    constexpr uint32_t element_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in = get_compile_time_arg_val(1);
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(2);
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(3);
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(4);
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(5);
    constexpr uint32_t stage_cb_id = get_compile_time_arg_val(6);
    constexpr auto dst_args = TensorAccessorArgs<7>();

    constexpr uint32_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;
    constexpr uint32_t face_height_width = FACE_HEIGHT * FACE_WIDTH;
    constexpr uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;
    constexpr uint32_t FACE_ROW_BYTES = FACE_WIDTH * element_size;
    constexpr uint32_t FACE_BYTES = face_height_width * element_size;

    const auto s = TensorAccessor(dst_args, dst_addr);

    cb_reserve_back(stage_cb_id, 2);
    const uint32_t stage_base0 = get_write_ptr(stage_cb_id);
    noc_async_read_one_packet_set_state(get_noc_addr(stage_base0), FACE_ROW_BYTES);
    bool slot_dirty[2] = {false, false};

    const uint32_t NtNt = Nt * Nt;
    const uint32_t NtS = Nt * S;
    const uint32_t CPG = Ct / S;
    uint32_t group = first_group;
    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        const uint32_t it = group / NtS;
        const uint32_t rem = group - it * NtS;
        const uint32_t jt = rem / S;
        const uint32_t sub = rem - jt * S;
        const uint32_t page_base = it * Nt + jt;
        const uint32_t row_base_g = it * TILE_HEIGHT;
        const uint32_t rows_valid = (row_base_g + TILE_HEIGHT <= D1) ? TILE_HEIGHT
                                                                    : (D1 - row_base_g);
        if (rows_valid < TILE_HEIGHT) {
            // A slot may still be in flight from the previous group's last write.
            noc_async_write_barrier();
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
        const uint32_t rows_lo = rows_valid < FACE_HEIGHT ? rows_valid : FACE_HEIGHT;
        const uint32_t rows_hi = rows_valid > FACE_HEIGHT ? rows_valid - FACE_HEIGHT : 0;

        for (uint32_t q = 0; q < CPG; ++q) {
            const uint32_t ct = sub * CPG + q;
            uint32_t out_page = page_base + ct * TILE_HEIGHT * NtNt;
            cb_wait_front(cb_id_in, TILE_HEIGHT);
            const uint32_t group_l1_base = get_read_ptr(cb_id_in);
            for (uint32_t c = 0; c < TILE_HEIGHT; ++c) {
                const uint32_t slot = c & 1u;
                const uint32_t stage_base = stage_base0 + slot * tile_bytes;
                if (slot_dirty[slot]) {
                    noc_async_writes_flushed();
                }
                const uint32_t src_c = group_l1_base
                                     + (c / FACE_HEIGHT) * NUM_FACES_W * FACE_BYTES
                                     + (c % FACE_HEIGHT) * FACE_ROW_BYTES;
                uint32_t s0 = src_c;
                uint32_t s1 = src_c + FACE_BYTES;
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
                    s0 = src_c + FACE_HEIGHT * tile_bytes;
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
                noc_async_read_barrier();
                noc_async_write(stage_base, s.get_noc_addr(out_page), tile_bytes);
                out_page += NtNt;
                slot_dirty[slot] = true;
            }
            noc_async_write_barrier();
            slot_dirty[0] = false;
            slot_dirty[1] = false;
            cb_pop_front(cb_id_in, TILE_HEIGHT);
        }

        group += group_stride;
        if (group == group_wrap_hi) {
            group = group_wrap_lo;
        }
    }
}
