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
// The bias half is writer_sparse_bias.cpp's walk, unchanged in substance: a group is one
// (head, tile-row) band, the band's 32 index rows are read once and walked with a per-row
// cursor as the tile-column window advances (idx is sorted ascending along K), each L1 slot is
// primed with the mask constant once and afterwards only ever REPAIRED by replaying the walk of
// the tile that last used it. The one difference is where the tile goes: into a circular buffer
// for the compute kernel instead of to DRAM. So the repair has to reclaim slots through
// cb_reserve_back -- a slot is only safe to touch once the compute kernel has consumed it -- and
// the reclaim must happen while THIS band's index rows are still the ones in L1, which is why
// the whole ring is drained at the end of every band.
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
    constexpr uint32_t cb_cur = get_compile_time_arg_val(4);   // reader-local scratch
    constexpr uint32_t It = get_compile_time_arg_val(5);
    constexpr uint32_t Jt = get_compile_time_arg_val(6);
    constexpr uint32_t Kt = get_compile_time_arg_val(7);
    constexpr uint32_t K = get_compile_time_arg_val(8);
    constexpr uint32_t L = get_compile_time_arg_val(9);
    constexpr uint32_t fill = get_compile_time_arg_val(10);    // fp32 bit pattern
    constexpr uint32_t SLOTS = get_compile_time_arg_val(11);   // cb_bias depth, power of two
    constexpr auto scores_args = TensorAccessorArgs<12>();
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
    volatile tt_l1_ptr uint32_t* slot_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cur));
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

    constexpr uint32_t NO_TILE = 0xFFFFFFFFu;
    uint32_t slot_jt[SLOTS];
    uint32_t slot_rows[SLOTS];
    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        slot_jt[slot] = NO_TILE;
        slot_rows[slot] = 0;
    }
    uint32_t cur[TILE_H];
    uint32_t pushed = 0;

    // Restore slot `slot` to all-fill by replaying the walk of the tile that last used it. Only
    // ever called with idx_l1 still holding the band that tile belonged to, and only after the
    // compute kernel has released the page.
    auto repair = [&](uint32_t slot) {
        if (slot_jt[slot] == NO_TILE) {
            return;
        }
        volatile tt_l1_ptr uint32_t* out_l1 =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias_base + slot * OUT_TILE_BYTES);
        volatile tt_l1_ptr uint32_t* sc = slot_l1 + slot * TILE_H;
        const uint32_t lo = slot_jt[slot] * 32;
        const uint32_t hi = lo + 32;
        const uint32_t nr = slot_rows[slot];
        for (uint32_t r = 0; r < nr; ++r) {
            volatile tt_l1_ptr uint32_t* irow = idx_l1 + r * K;
            for (uint32_t sidx = sc[r]; sidx < K; ++sidx) {
                const uint32_t j = irow[sidx];
                if (j >= hi) {
                    break;
                }
                out_l1[face_off(r, j - lo)] = fill;
            }
        }
        slot_jt[slot] = NO_TILE;
    };

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

            cb_reserve_back(cb_bias, 1);
            const uint32_t slot = pushed & (SLOTS - 1);
            volatile tt_l1_ptr uint32_t* out_l1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias_base + slot * OUT_TILE_BYTES);
            repair(slot);

            volatile tt_l1_ptr uint32_t* sc = slot_l1 + slot * TILE_H;
            const uint32_t col_lo = jt * 32;
            const uint32_t col_hi = col_lo + 32;
            for (uint32_t r = 0; r < rows; ++r) {
                volatile tt_l1_ptr uint32_t* irow = idx_l1 + r * K;
                uint32_t sidx = cur[r];
                sc[r] = sidx;
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
            slot_jt[slot] = jt;
            slot_rows[slot] = rows;

            noc_async_read_barrier();
            cb_push_back(cb_scores, 1);
            cb_push_back(cb_bias, 1);
            ++pushed;
        }

        // Reclaim the whole ring before the band's index rows are overwritten: reserving every
        // page waits until the compute kernel has consumed all of them, and the replay is only
        // correct against the index rows of the band that wrote them.
        cb_reserve_back(cb_bias, SLOTS);
        for (uint32_t slot = 0; slot < SLOTS; ++slot) {
            repair(slot);
        }
    }
}
