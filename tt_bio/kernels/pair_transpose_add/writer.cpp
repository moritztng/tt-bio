// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// pair_transpose_add writer: z[J, I] += u[I, J] for one 32 x 32 tile block per unit. The reader is
// pair_transpose's split reader, unchanged: it brings the unit's 32 input tiles and shuffles output
// tiles 0..QR-1 into the slot. This kernel shuffles QR..31 the same way, reads the 32 z tiles the
// unit lands on, and hands both to the compute kernel instead of writing the slot out; the sums come
// back in c_17 and go to the z pages they were read from. Each page belongs to one unit, so it is
// read before it is written.
//
// The slot CB (c_16, two 32-tile slots) is written by raw address as in pair_transpose; here it is
// also a compute input, so this kernel reserves and pushes it (32 pages, slot = unit parity, the same
// order the CB pointer walks). c_2 tells the reader a slot is free again: compute pops c_16 before it
// pushes the unit's last sums, so once all 32 sums are here the slot is no longer read.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t z_addr = get_common_arg_val<uint32_t>(0);
    constexpr uint32_t cb_in = 0, cb_half = 1, cb_free = 2, cb_out = 16, cb_z = 3, cb_sum = 17;
    constexpr uint32_t S1t = get_compile_time_arg_val(0);
    constexpr uint32_t S2t = get_compile_time_arg_val(1);
    constexpr uint32_t Ct = get_compile_time_arg_val(2);
    constexpr uint32_t ROW = get_compile_time_arg_val(3);
    constexpr uint32_t QR = get_compile_time_arg_val(4);
    constexpr uint32_t GRAN = get_compile_time_arg_val(5);
    constexpr auto z_args = TensorAccessorArgs<6>();
    const auto z = TensorAccessor(z_args, z_addr);

    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t num = get_arg_val<uint32_t>(1);
    const uint32_t tile_bytes = get_tile_size(cb_out);
    constexpr uint32_t FACE = 16 * ROW;
    const uint32_t out_base = get_write_ptr(cb_out);

    for (uint32_t i = 0; i < num; ++i) {
        const uint32_t u = first + i;
        const uint32_t ct = u % Ct;
        const uint32_t It = (u / Ct) % S1t;
        const uint32_t Jt = u / (Ct * S1t);
        auto page = [&](uint32_t q) { return ((Jt * 32 + q) * S1t + It) * Ct + ct; };

        // z tiles of this unit, while the shuffle below runs on the same NoC
        cb_reserve_back(cb_z, 32);
        uint32_t lz = get_write_ptr(cb_z);
        for (uint32_t q = 0; q < 32; ++q, lz += tile_bytes) {
            noc_async_read(z.get_noc_addr(page(q)), lz, tile_bytes);
        }

        cb_reserve_back(cb_out, 32);
        const uint32_t slot = out_base + (i & 1) * 32 * tile_bytes;
        cb_wait_front(cb_in, 32);
        const uint32_t in = get_read_ptr(cb_in);
        for (uint32_t r = 0; r < 32; ++r) {
            const uint32_t src_tile = in + r * tile_bytes;
            const uint32_t dst_row = ((r >> 4) * 2) * FACE + (r & 15) * ROW;
            for (uint32_t q = QR; q < 32; ++q) {
                const uint32_t src = src_tile + ((q >> 4) * 2) * FACE + (q & 15) * ROW;
                const uint32_t dst = slot + q * tile_bytes + dst_row;
                noc_async_read(get_noc_addr(src), dst, ROW);
                noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_z, 32);
        cb_wait_front(cb_half, 1);                  // reader's rows are in the slot, its input read done
        cb_pop_front(cb_half, 1);
        cb_pop_front(cb_in, 32);
        cb_push_back(cb_out, 32);

        for (uint32_t q = 0; q < 32; q += GRAN) {
            cb_wait_front(cb_sum, GRAN);
            uint32_t l1 = get_read_ptr(cb_sum);
            for (uint32_t k = 0; k < GRAN; ++k, l1 += tile_bytes) {
                noc_async_write(l1, z.get_noc_addr(page(q + k)), tile_bytes);
            }
            noc_async_writes_flushed();
            cb_pop_front(cb_sum, GRAN);
        }
        if (i + 2 < num) {
            cb_reserve_back(cb_free, 1);
            cb_push_back(cb_free, 1);
        }
    }
    noc_async_write_barrier();
}
