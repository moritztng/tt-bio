// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 fused attention-score reader: streams the bf16 scores and BUILDS the fp32 sparse bias
// tile-by-tile in L1, so the [1, H, L, N] bias tensor is never materialised in DRAM at all.
//
// Together with compute_fused_scores.cpp and writer_fused_scores.cpp this replaces five ops:
//     full(-1e4, bf16) -> scatter(dim=3, idx, pair_bias) -> typecast(fp32)
//     -> typecast(scores, fp32) -> add(scores * scale, bias)
// with one pass reading 90.3 MB and writing 180.6 MB at the production shape.
//
// The bias half is writer_sparse_bias.cpp's walk: a group is one (head, tile-row) band, the band's
// 32 index rows are read once and walked with a per-row cursor as the tile-column window advances
// (idx is sorted ascending along K), and the tile goes into a circular buffer for the compute kernel
// instead of to DRAM.
//
// A used L1 slot returns to all-fill by ONE local 4 KB L1->L1 copy from a pristine template page.
// writer_sparse_bias.cpp instead replayed the tile's walk writing the mask constant back, which is
// ~120 scalar L1 accesses per tile; the copy measured 1.686 -> 1.393 ms/call against it at the
// production shape, bit-exact, and it also removes the per-band ring reclaim, the coupling that
// forced that reclaim to happen while the band's index rows were still in L1, and 32 cursor stores
// per tile. The invariant is "every free slot is pristine": true at entry, and each tile restores the
// NEXT slot while poking the current one, so the read barrier already in the loop covers it.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known device-wedge
// cause in this codebase).
#include "api/dataflow/dataflow_api.h"

// Element offset of (r, c) inside a 32x32 tile laid out as four 16x16 faces.
inline uint32_t face_off(uint32_t r, uint32_t c) {
    return ((((r >> 4) << 1) | (c >> 4)) << 8) + ((r & 15) << 4) + (c & 15);
}

void kernel_main() {
    const uint32_t scores_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t pb_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(2);

    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_scores = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(2);   // reader-local scratch
    constexpr uint32_t cb_pb = get_compile_time_arg_val(3);    // reader-local scratch
    constexpr uint32_t cb_tpl = get_compile_time_arg_val(4);   // the pristine all-fill page
    constexpr uint32_t It = get_compile_time_arg_val(5);
    constexpr uint32_t Jt = get_compile_time_arg_val(6);
    constexpr uint32_t Kt = get_compile_time_arg_val(7);
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t L = get_compile_time_arg_val(9);
    constexpr uint32_t fill = get_compile_time_arg_val(10);    // fp32 bit pattern
    constexpr uint32_t SLOTS = get_compile_time_arg_val(11);   // cb_bias depth, power of two
    // Diagnostic only, never set in production: skip the poke walk and the repair replay, keeping
    // every read, every CB handshake, the compute pass and the write, so the op splits into "the
    // pipeline" and "the per-element placement". That decomposition is what refuted L6c.
    constexpr uint32_t NOPOKE = get_compile_time_arg_val(12);
    constexpr auto scores_args = TensorAccessorArgs<13>();
    constexpr auto pb_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<pb_args.next_compile_time_args_offset()>();

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t SCORES_TILE_BYTES = 32 * 32 * 2;
    constexpr uint32_t BIAS_TILE_BYTES = 32 * 32 * 2;
    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * 4;
    constexpr uint32_t OUT_TILE_WORDS = 32 * 32;

    const auto s_scores = TensorAccessor(scores_args, scores_addr);
    const auto s_pb = TensorAccessor(pb_args, pb_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr);

    volatile tt_l1_ptr uint32_t* idx_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_idx));
    const uint32_t pb_base = get_write_ptr(cb_pb);
    volatile tt_l1_ptr uint16_t* pb_l1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pb_base);
    // The ring base. Captured before any push, so page `s` of the ring is bias_base + s*4096 and
    // the slot the n-th push lands in is n % SLOTS -- the same bookkeeping the DRAM version did
    // with jt, and it must agree with the CB's own pointer, which advances one page per push.
    const uint32_t bias_base = get_write_ptr(cb_bias);

    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias_base + slot * OUT_TILE_BYTES);
        for (uint32_t w = 0; w < OUT_TILE_WORDS; ++w) {
            p[w] = fill;
        }
    }
    const uint32_t tpl_addr = get_write_ptr(cb_tpl);
    {
        volatile tt_l1_ptr uint32_t* p = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tpl_addr);
        for (uint32_t w = 0; w < OUT_TILE_WORDS; ++w) {
            p[w] = fill;
        }
    }
    const uint64_t tpl_noc = get_noc_addr(tpl_addr);

    uint32_t cur[TILE_H];
    uint32_t pushed = 0;


    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t h = group / It;
        const uint32_t it = group - h * It;
        const uint32_t row_base = it * TILE_H;
        const uint32_t rows = (row_base + TILE_H <= L) ? TILE_H : (L - row_base);

        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(s_idx.get_noc_addr(row_base + r),
                           reinterpret_cast<uint32_t>(idx_l1) + r * K * 4, K * 4);
        }
        const uint32_t pb_page0 = (h * It + it) * Kt;
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            noc_async_read(s_pb.get_noc_addr(pb_page0 + kt), pb_base + kt * BIAS_TILE_BYTES,
                           BIAS_TILE_BYTES);
        }
        noc_async_read_barrier();

        for (uint32_t r = 0; r < rows; ++r) {
            cur[r] = 0;
        }

        const uint32_t tile_page0 = (h * It + it) * Jt;
        for (uint32_t jt = 0; jt < Jt; ++jt) {
            // The scores tile is issued FIRST and barriered last, so its DRAM latency hides
            // behind this tile's poke walk instead of adding to it.
            cb_reserve_back(cb_scores, 1);
            noc_async_read(s_scores.get_noc_addr(tile_page0 + jt), get_write_ptr(cb_scores),
                           SCORES_TILE_BYTES);

            // The next slot is reserved as well, because this tile restores it.
            cb_reserve_back(cb_bias, 2);
            const uint32_t slot = pushed & (SLOTS - 1);
            volatile tt_l1_ptr uint32_t* out_l1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias_base + slot * OUT_TILE_BYTES);
            const uint32_t col_lo = jt * 32;
            const uint32_t col_hi = col_lo + 32;
            for (uint32_t r = 0; NOPOKE ? false : r < rows; ++r) {
                volatile tt_l1_ptr uint32_t* irow = idx_l1 + r * K;
                uint32_t sidx = cur[r];
                while (sidx < K) {
                    const uint32_t j = irow[sidx];
                    if (j >= col_hi) {
                        break;
                    }
                    const uint32_t bv =
                        pb_l1[(sidx >> 5) * (BIAS_TILE_BYTES / 2) + face_off(r, sidx & 31)];
                    out_l1[face_off(r, j - col_lo)] = bv << 16;
                    ++sidx;
                }
                cur[r] = sidx;
            }
            // Restore the slot this core will poke next. It is free (reserved above) and dirty
            // from its previous use, so this one transaction is the whole repair.
            noc_async_read(tpl_noc, bias_base + ((pushed + 1) & (SLOTS - 1)) * OUT_TILE_BYTES,
                           OUT_TILE_BYTES);
            noc_async_read_barrier();
            cb_push_back(cb_scores, 1);
            cb_push_back(cb_bias, 1);
            ++pushed;
        }

    }
}
