// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 sparse attention bias, built in one pass.
//
// Replaces, for a [1, H, L, N] fp32 attention bias with N = align_tile(L):
//     ttnn.full(-1e4, bf16) -> ttnn.scatter(dim=3, idx, pair_bias) -> ttnn.typecast(fp32)
// with a single kernel that materialises the fp32 result directly:
//     out[h, i, j] = fp32(pair_bias[h, i, s])  if j == idx[i, s] for some s in [0, K)
//                    fill                      otherwise
// `fill` is passed in as a bit pattern, so it is exactly whatever the bf16 template
// held after its widen (bf16(-1e4) = -9984.0f) rather than a re-derived constant.
//
// Bit-exact by construction: widening bf16 to fp32 is a 16-bit left shift, the values
// written are the same values in the same positions, and nothing is recomputed.
//
// WHY THIS IS FASTER. `ttnn.scatter` is out-of-place: it copies all H*L*N elements and
// then writes the H*L*K it was given, and its copy runs at 9.6 G elem/s where a clone
// runs at 92 (per-element rate limited, not bandwidth limited -- see
// state/rfd3-host-half.md §3). This kernel never reads the H*L*N side at all: it
// generates each output tile in L1 from a template and pokes only the indices that fall
// inside that tile's 32 columns, so the only DRAM traffic is the one fp32 write it owes
// plus the K-wide bias and index rows. That is 271 MB of unavoidable traffic against
// scatter's 180 MB at a tenth of the rate.
//
// WORK SPLIT. One group = one (head, tile-row) band, so H * It groups, each producing
// the Jt output tiles of one 32-row strip. A core owns a contiguous range of groups.
// The 32 index rows of a band are read once and walked with a per-row cursor as jt
// advances, which is what makes the placement a single forward merge: idx is sorted
// ascending along K (`_create_attention_indices` ends in torch.sort), so a row's entries
// enter the tile column window in order and each is visited exactly once per band.
// Total pokes are H*L*K = 1.72 M at the production shape, not H*L*N = 45 M.
//
// THE TEMPLATE IS NOT RE-COPIED PER TILE. Each of the OUT_SLOTS L1 pages is filled with
// `fill` once at kernel entry and then kept: before a slot is reused, the positions the
// previous tile poked are restored by REPLAYING that tile's cursor walk -- the same merge,
// writing `fill` instead of the bias. So the repair needs no record of what was written,
// only the 32 cursor values the tile started from (128 bytes per slot), and it is exact by
// duality rather than by a bound on how many pokes a tile can hold. That is what lets
// OUT_SLOTS be large: a per-tile 4 KB L1 refill, or an undo list of offsets, would both
// cost L1 or time per slot.
//
//
// BATCH. A group is a (design, head, tile-row) band and the group index is that triple flattened,
// so every page this kernel touches except the neighbour index is still `group * <tiles per band>`
// -- tile pages of a [B, H, L, *] tensor run in exactly that order. The index is the one tensor
// that is NOT per head: it is [B, 1, L, K] ROW_MAJOR, one page per row, so its page needs the design
// back out of the group, which is the only reason HIt is passed in. Multiplicity batching is the
// default for a multi-design run (`--batch_size 8`), so batch > 1 is the common path, not an edge.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"

// Element offset of (r, c) inside a 32x32 tile laid out as four 16x16 faces.
inline uint32_t face_off(uint32_t r, uint32_t c) {
    return ((((r >> 4) << 1) | (c >> 4)) << 8) + ((r & 15) << 4) + (c & 15);
}

void kernel_main() {
    // The three buffer addresses are the only values that change between calls at a fixed
    // shape/grid, so they live in the common runtime args and the host caches the whole
    // ProgramDescriptor (a per-call rebuild of the per-core args costs more than the op).
    const uint32_t bias_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t idx_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t out_addr = get_common_arg_val<uint32_t>(2);

    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_idx = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_out = get_compile_time_arg_val(2);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(3);
    constexpr uint32_t It = get_compile_time_arg_val(4);        // tile rows of the output
    constexpr uint32_t Jt = get_compile_time_arg_val(5);        // tile cols of the output
    constexpr uint32_t Kt = get_compile_time_arg_val(6);        // tile cols of the bias
    constexpr uint32_t K = get_compile_time_arg_val(7);         // logical keys per row
    constexpr uint32_t L = get_compile_time_arg_val(8);         // logical rows
    constexpr uint32_t fill = get_compile_time_arg_val(9);      // fp32 bit pattern
    constexpr uint32_t OUT_SLOTS = get_compile_time_arg_val(10);
    constexpr uint32_t HIt = get_compile_time_arg_val(11);   // heads * tile rows, one design's bands
    constexpr auto bias_args = TensorAccessorArgs<12>();
    constexpr auto idx_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t IDX_ROW_WORDS = K;
    constexpr uint32_t BIAS_TILE_BYTES = 32 * 32 * 2;
    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * 4;
    constexpr uint32_t OUT_TILE_WORDS = 32 * 32;

    const auto s_bias = TensorAccessor(bias_args, bias_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);

    volatile tt_l1_ptr uint32_t* idx_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_idx));
    const uint32_t bias_base = get_write_ptr(cb_bias);
    volatile tt_l1_ptr uint16_t* bias_l1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bias_base);
    const uint32_t out_base = get_write_ptr(cb_out);
    // Per-slot entry cursors: 32 words per slot, the state the repair replay needs.
    volatile tt_l1_ptr uint32_t* slot_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_slot));

    // Prime every output slot with the fill value once. From here on a slot is only ever
    // repaired, never refilled.
    for (uint32_t slot = 0; slot < OUT_SLOTS; ++slot) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_base + slot * OUT_TILE_BYTES);
        for (uint32_t w = 0; w < OUT_TILE_WORDS; ++w) {
            p[w] = fill;
        }
    }

    constexpr uint32_t NO_TILE = 0xFFFFFFFFu;
    uint32_t slot_jt[OUT_SLOTS];
    uint32_t slot_rows[OUT_SLOTS];
    for (uint32_t slot = 0; slot < OUT_SLOTS; ++slot) {
        slot_jt[slot] = NO_TILE;
        slot_rows[slot] = 0;
    }
    uint32_t cur[TILE_H];

    const uint32_t end_group = start_group + num_groups;
    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t b = group / HIt;
        const uint32_t it = group % It;
        const uint32_t row_base = it * TILE_H;
        const uint32_t rows = (row_base + TILE_H <= L) ? TILE_H : (L - row_base);

        // The index rows of this band. idx is ROW_MAJOR uint32 [1, 1, L, K] -- one page per
        // row -- and is shared by every head, so a band's pages are read once per (h, it).
        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(s_idx.get_noc_addr(b * L + row_base + r),
                           reinterpret_cast<uint32_t>(idx_l1) + r * IDX_ROW_WORDS * 4,
                           IDX_ROW_WORDS * 4);
        }
        // The band's bias tiles: [1, H, L, K] bf16 TILE, page (h*It + it)*Kt + kt.
        const uint32_t bias_page0 = group * Kt;
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            noc_async_read(s_bias.get_noc_addr(bias_page0 + kt),
                           bias_base + kt * BIAS_TILE_BYTES, BIAS_TILE_BYTES);
        }
        noc_async_read_barrier();

        for (uint32_t r = 0; r < rows; ++r) {
            cur[r] = 0;
        }

        // Restore slot `slot` to all-fill by replaying the walk of the tile that last used
        // it. Only ever called with the slot's write already flushed and with idx_l1 still
        // holding the band that tile belonged to.
        auto repair = [&](uint32_t slot) {
            if (slot_jt[slot] == NO_TILE) {
                return;
            }
            volatile tt_l1_ptr uint32_t* out_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(
                out_base + slot * OUT_TILE_BYTES);
            volatile tt_l1_ptr uint32_t* sc = slot_l1 + slot * TILE_H;
            const uint32_t lo = slot_jt[slot] * 32;
            const uint32_t hi = lo + 32;
            const uint32_t nr = slot_rows[slot];
            for (uint32_t r = 0; r < nr; ++r) {
                volatile tt_l1_ptr uint32_t* irow = idx_l1 + r * IDX_ROW_WORDS;
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

        uint32_t out_page = group * Jt;
        for (uint32_t jt = 0; jt < Jt; ++jt) {
            const uint32_t slot = jt & (OUT_SLOTS - 1);
            const uint32_t out_addr_l1 = out_base + slot * OUT_TILE_BYTES;
            volatile tt_l1_ptr uint32_t* out_l1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_addr_l1);
            repair(slot);

            volatile tt_l1_ptr uint32_t* sc = slot_l1 + slot * TILE_H;
            const uint32_t col_lo = jt * 32;
            const uint32_t col_hi = col_lo + 32;
            for (uint32_t r = 0; r < rows; ++r) {
                volatile tt_l1_ptr uint32_t* irow = idx_l1 + r * IDX_ROW_WORDS;
                uint32_t sidx = cur[r];
                sc[r] = sidx;
                while (sidx < K) {
                    const uint32_t j = irow[sidx];
                    if (j >= col_hi) {
                        break;
                    }
                    const uint32_t bv =
                        bias_l1[(sidx >> 5) * (BIAS_TILE_BYTES / 2) + face_off(r, sidx & 31)];
                    out_l1[face_off(r, j - col_lo)] = bv << 16;
                    ++sidx;
                }
                cur[r] = sidx;
            }
            slot_jt[slot] = jt;
            slot_rows[slot] = rows;

            noc_async_write(out_addr_l1, s_out.get_noc_addr(out_page), OUT_TILE_BYTES);
            ++out_page;
            // Every slot is free again after this, so the reuse at jt + 1 can repair in place.
            if (slot == OUT_SLOTS - 1) {
                noc_async_write_barrier();
            }
        }
        // Jt is not in general a multiple of OUT_SLOTS. Drain, then hand the next band a set
        // of clean slots -- the replay has to happen while THIS band's index rows are still
        // the ones in L1.
        noc_async_write_barrier();
        for (uint32_t slot = 0; slot < OUT_SLOTS; ++slot) {
            repair(slot);
        }
    }
}
