// SPDX-FileCopyrightText: (c) 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// triangle attention tail of the ending block, into z^T: z[s, b] += u[b, s] without writing u. A unit is
// an R x 32 block (bg, st), R = 32, 16 or 8 b rows: compute leaves its R row tiles (bg*R + r, st), Nt
// channel tiles each, in c_16 ([r][n], all R*Nt pages). Per channel tile n this kernel reads rows
// r0 .. r0+R-1 (r0 = bg*R % 32) of the 32 z tiles the block lands on (z page (s*Bt + bt)*Nt + n,
// bt = bg*R / 32, s = st*32 + q) into c_3, shuffles row q of staged tile (r, n) into row r0 + r of tile q
// of c_18 (pair_transpose_add's shuffle, two 32 B face rows per move), and writes those rows of the 32
// sums compute returns in c_17 back where they were read. Each z row belongs to one unit, so units that
// share a tile (R < 32) never write each other's rows; the rows a unit does not own are summed and dropped.
// R < 32 halves or quarters the stage, which is what lets a wide pair (320, 448 channels) fit in L1.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t Nt = get_compile_time_arg_val(0);
    constexpr uint32_t St = get_compile_time_arg_val(1);
    constexpr uint32_t Bt = get_compile_time_arg_val(2);
    constexpr uint32_t GRAN = get_compile_time_arg_val(3);
    constexpr uint32_t R = get_compile_time_arg_val(4);
    constexpr auto z_args = TensorAccessorArgs<5>();
    const auto z = TensorAccessor(z_args, get_common_arg_val<uint32_t>(0));
    const uint32_t u0 = get_arg_val<uint32_t>(0) / R;
    const uint32_t nunits = get_arg_val<uint32_t>(1) / R;

    constexpr uint32_t stage_cb = tt::CBIndex::c_16, z_cb = tt::CBIndex::c_3;
    constexpr uint32_t t_cb = tt::CBIndex::c_18, s_cb = tt::CBIndex::c_17;
    constexpr uint32_t ROW = 16 * 2, FACE = 16 * ROW;
    const uint32_t tb = get_tile_size(stage_cb);
    // rows r0 .. r0+R-1 of a tile: the whole tile, else one R-row run in each face of the pair holding r0
    constexpr uint32_t SEG = R == 32 ? 0 : R * ROW;
    auto rows = [&](uint64_t far, uint32_t near, uint32_t off, bool write) {
        if constexpr (SEG == 0) {
            if (write) noc_async_write(near, far, 4 * FACE);
            else noc_async_read(far, near, 4 * FACE);
        } else {
            for (uint32_t f = 0; f < 2; ++f) {
                if (write) noc_async_write(near + off + f * FACE, far + off + f * FACE, SEG);
                else noc_async_read(far + off + f * FACE, near + off + f * FACE, SEG);
            }
        }
    };

    for (uint32_t u = u0; u < u0 + nunits; ++u) {
        const uint32_t bg = u / St, st = u % St, bt = bg * R / 32, r0 = bg * R % 32;
        const uint32_t off = (r0 >> 4) * 2 * FACE + (r0 & 15) * ROW;
        auto page = [&](uint32_t q, uint32_t n) { return ((st * 32 + q) * Bt + bt) * Nt + n; };
        // QSPREAD: pass n moves channel tile (n + q) % Nt for z row tile q, so the 32 pages of a pass, Bt*Nt apart
        // (a multiple of the 8 DRAM banks at every Mol-1 width), spread over the banks. Each (q, channel) still moves
        // exactly once per unit, and compute pairs c_3 and c_18 tiles as they arrive, so the sums are the same.
        auto ch = [&](uint32_t n, uint32_t q) -> uint32_t {
#ifdef QSPREAD
            return (n + q) % Nt;
#else
            return n;
#endif
        };
        cb_wait_front(stage_cb, R * Nt);
        const uint32_t stage = get_read_ptr(stage_cb);
        for (uint32_t n = 0; n < Nt; ++n) {
            cb_reserve_back(z_cb, 32);
            uint32_t lz = get_write_ptr(z_cb);
            for (uint32_t q = 0; q < 32; ++q, lz += tb) rows(z.get_noc_addr(page(q, ch(n, q))), lz, off, false);
            cb_reserve_back(t_cb, 32);
            const uint32_t slot = get_write_ptr(t_cb);
            for (uint32_t r = 0; r < R; ++r) {
                const uint32_t rr = r0 + r;
                const uint32_t dst_row = ((rr >> 4) * 2) * FACE + (rr & 15) * ROW;
                for (uint32_t q = 0; q < 32; ++q) {
                    const uint32_t src_tile = stage + (r * Nt + ch(n, q)) * tb;
                    const uint32_t src = src_tile + ((q >> 4) * 2) * FACE + (q & 15) * ROW;
                    const uint32_t dst = slot + q * tb + dst_row;
                    noc_async_read(get_noc_addr(src), dst, ROW);
                    noc_async_read(get_noc_addr(src + FACE), dst + FACE, ROW);
                }
            }
            noc_async_read_barrier();
            cb_push_back(z_cb, 32);
            cb_push_back(t_cb, 32);
            for (uint32_t q = 0; q < 32; q += GRAN) {
                cb_wait_front(s_cb, GRAN);
                uint32_t l1 = get_read_ptr(s_cb);
                for (uint32_t k = 0; k < GRAN; ++k, l1 += tb) rows(z.get_noc_addr(page(q + k, ch(n, q + k))), l1, off, true);
                noc_async_writes_flushed();
                cb_pop_front(s_cb, GRAN);
            }
        }
        cb_pop_front(stage_cb, R * Nt);
    }
    noc_async_write_barrier();
}
