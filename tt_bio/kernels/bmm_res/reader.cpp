// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// bmm_res reader (tt_bio/bmm_res.py): out[c] = a[c] @ b[c] with b[c] resident in L1 for the whole channel.
// Per channel, b[c]'s KT x NT tiles go into c_1 in one reservation (c_1 holds one or two channels, so with two the
// next b[c] loads while compute still works through the current one); then a[c]'s MT row tiles stream through c_0,
// KT pages each. Row tile r of channel c is pages [(c*MT + r)*KT, +KT) of a, so every byte of a, b and out moves
// once and no core reads another core's channel.
// TA: a is stored [K, M], so row tile r is the column of pages c*MT*KT + k*MT + r, read into c_2 for compute to
// transpose. TB: b is stored [N, K], so its tile (k, n) is page n*KT + k; c_1 still holds b in (k, n) order and the
// matmul unpacker transposes each tile.
//
// Split (NB < NT): a work unit is channel u / NS with output columns [j*NB, (j+1)*NB), j = u % NS; c_1 holds that
// column block of b[c], and a[c] streams once per block. The transposed a moves in AG-tile groups (compute's own
// grouping), so c_2 needs only A_DEPTH groups; unsplit, AG = KT.
//
// B8 (BG > 0): b moves through c_3 in BG-tile groups, in the same (k, n) order, for compute to repack into c_1 as
// bfp8 (bmm_res.BRES_B8).
//
// Common runtime args: a, b addresses. Per-core runtime args: first unit, unit count.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t a_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t b_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t first = get_arg_val<uint32_t>(0);
    const uint32_t count = get_arg_val<uint32_t>(1);
    constexpr uint32_t MT = get_compile_time_arg_val(0);
    constexpr uint32_t KT = get_compile_time_arg_val(1);
    constexpr uint32_t NT = get_compile_time_arg_val(2);
    constexpr bool TA = get_compile_time_arg_val(3);
    constexpr bool TB = get_compile_time_arg_val(4);
    constexpr uint32_t NB = get_compile_time_arg_val(5);
    constexpr uint32_t AG = get_compile_time_arg_val(6);
    constexpr uint32_t BG = get_compile_time_arg_val(7);
    constexpr uint32_t NS = NT / NB;
    constexpr uint32_t cb_a = TA ? 2 : 0, cb_b = BG ? 3 : 1;
    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto b_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();
    const auto sa = TensorAccessor(a_args, a_addr);
    const auto sb = TensorAccessor(b_args, b_addr);
    const uint32_t ta = get_tile_size(cb_a), tb = get_tile_size(cb_b);

    for (uint32_t u = first; u < first + count; ++u) {
        const uint32_t c = u / NS, n0 = (u % NS) * NB;
        constexpr uint32_t G = BG ? BG : KT * NB;
        uint32_t w = 0, i = 0;
        for (uint32_t k = 0; k < KT; ++k) {
            for (uint32_t n = n0; n < n0 + NB; ++n, w += tb) {
                if (i == 0) {
                    cb_reserve_back(cb_b, G);
                    w = get_write_ptr(cb_b);
                }
                noc_async_read_page(c * KT * NT + (TB ? n * KT + k : k * NT + n), sb, w);
                if (++i == G) {
                    noc_async_read_barrier();
                    cb_push_back(cb_b, G);
                    i = 0;
                }
            }
        }
        for (uint32_t r = 0; r < MT; ++r) {
            for (uint32_t k0 = 0; k0 < KT; k0 += AG) {
                const uint32_t g = KT - k0 < AG ? KT - k0 : AG;
                cb_reserve_back(cb_a, AG);
                w = get_write_ptr(cb_a);
#ifndef BRES_NO_A  // time attribution only (af2_trimul_bmm_sweep --abl)
                for (uint32_t k = k0; k < k0 + g; ++k, w += ta) {
                    noc_async_read_page(c * MT * KT + (TA ? k * MT + r : r * KT + k), sa, w);
                }
#endif
                noc_async_read_barrier();
                cb_push_back(cb_a, AG);  // a whole slot even when g < AG (compute's SLOT)
            }
        }
    }
}
