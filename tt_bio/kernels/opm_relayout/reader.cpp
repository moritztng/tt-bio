// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// opm_relayout reader. The OPM contraction with b's rows in (j d) order gives z = [(i c), (j d)],
// whose tile (i, j) is pair (i, j)'s [c, d] block. The projection wants y = [(i j), (c d)]. With
// C = D = 32 both are groups of 32 consecutive pages: group g = i * J/32 + jt owns input pages
// 32g + jl (tiles (i, 32 jt + jl)) and output pages 32g + c, and
//   out_tile_c[jl, :] = in_tile_jl[c, :].
// That is reblock_permute_back's gather with no transpose after it, so this is its reader with the
// page formula changed: 32 aligned 2 KB DRAM reads into a scratch window, then 64 L1 -> L1 32-byte
// face-row copies per output tile. Pure data movement, so the result is bit-exact.
//
// The one-packet state is re-armed per group: the DRAM page reads share the read command buffer
// (reblock_permute_back's reader documents both silent failure modes).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t first_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);

    constexpr uint32_t scratch_cb = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();

    constexpr uint32_t TILES = 32;
    constexpr uint32_t TILE_BYTES = 2048;   // bf16 32 x 32
    constexpr uint32_t FACE_BYTES = 512;    // 16 x 16 bf16
    constexpr uint32_t ROW_BYTES = 32;      // one face row
    constexpr uint32_t FACE_H = 16;

    const auto s = TensorAccessor(src_args, src_addr);

    cb_reserve_back(scratch_cb, TILES);
    const uint32_t base = get_write_ptr(scratch_cb);
    const uint64_t state = get_noc_addr(base);

    for (uint32_t g = first_group; g < first_group + num_groups; ++g) {
        uint32_t page = g * TILES;
        uint32_t dst = base;
        for (uint32_t jl = 0; jl < TILES; ++jl) {
            noc_async_read_page(page + jl, s, dst);
            dst += TILE_BYTES;
        }
        noc_async_read_barrier();
        noc_async_read_one_packet_set_state(state, ROW_BYTES);

        cb_reserve_back(out_cb, TILES);
        const uint32_t out = get_write_ptr(out_cb);
        for (uint32_t c = 0; c < TILES; ++c) {
            // row c of every source tile: face pair (c / 16), row (c % 16) inside it
            uint32_t s0 = base + (c / FACE_H) * 2 * FACE_BYTES + (c % FACE_H) * ROW_BYTES;
            uint32_t s1 = s0 + FACE_BYTES;
            const uint32_t t = out + c * TILE_BYTES;
            // destination rows jl < 16 sit in faces 0 / 1, rows 16..31 in faces 2 / 3
            uint32_t d0 = t;
            uint32_t d1 = t + FACE_BYTES;
            for (uint32_t jl = 0; jl < FACE_H; ++jl) {
                noc_async_read_one_packet_with_state(s0, d0);
                noc_async_read_one_packet_with_state(s1, d1);
                s0 += TILE_BYTES;
                s1 += TILE_BYTES;
                d0 += ROW_BYTES;
                d1 += ROW_BYTES;
            }
            d0 = t + 2 * FACE_BYTES;
            d1 = t + 3 * FACE_BYTES;
            for (uint32_t jl = FACE_H; jl < TILES; ++jl) {
                noc_async_read_one_packet_with_state(s0, d0);
                noc_async_read_one_packet_with_state(s1, d1);
                s0 += TILE_BYTES;
                s1 += TILE_BYTES;
                d0 += ROW_BYTES;
                d1 += ROW_BYTES;
            }
        }
        noc_async_read_barrier();
        cb_push_back(out_cb, TILES);
    }
}
