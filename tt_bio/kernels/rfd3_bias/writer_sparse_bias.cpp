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
// SLOTS ARE REFILLED BY THE NOC, NOT BY THE RISC. Every output slot starts as `fill`; once a
// band of slots has been written out, each used slot is restored with one local L1 -> L1 NOC copy
// of the all-fill tile, issued for every used slot at once behind a single barrier. The RISC used to
// replay each tile's cursor walk writing `fill` back (exact by duality), which doubled its L1 work:
// on Wormhole that work, not the DRAM write, was the op's cost (rfd3_bias.DIAG, gap_maskw.py).
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

// Bench-only diagnostics (rfd3_bias.DIAG): 1 every tile from the fill tile, 2 no DRAM writes, 3 = 1 without the scan.
#ifndef MASKW_DIAG
#define MASKW_DIAG 0
#endif
template <typename A>
inline void out_write(uint32_t l1, const A& s, uint32_t page, uint32_t bytes) {
    if constexpr (MASKW_DIAG != 2) {
        noc_async_write(l1, s.get_noc_addr(page), bytes);
    }
}

// Element offset of (r, c) inside a 32x32 tile laid out as four 16x16 faces.
inline uint32_t face_off(uint32_t r, uint32_t c) {
    return ((((r >> 4) << 1) | (c >> 4)) << 8) + ((r & 15) << 4) + (c & 15);
}

// Store one element of an output tile at element offset `off`. `v32` is the fp32 bit pattern; the
// bf16 form is its top half, which is exact for every value written here (all are widened bf16).
template <uint32_t OUT16>
inline void put(uint32_t tile_addr, uint32_t off, uint32_t v32) {
    if constexpr (OUT16) {
        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(tile_addr)[off] = static_cast<uint16_t>(v32 >> 16);
    } else {
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(tile_addr)[off] = v32;
    }
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
    constexpr uint32_t It = get_compile_time_arg_val(3);        // tile rows of the output
    constexpr uint32_t Jt = get_compile_time_arg_val(4);        // tile cols of the output
    constexpr uint32_t Kt = get_compile_time_arg_val(5);        // tile cols of the bias
    constexpr uint32_t K = get_compile_time_arg_val(6);         // logical keys per row
    constexpr uint32_t L = get_compile_time_arg_val(7);         // logical rows
    constexpr uint32_t fill = get_compile_time_arg_val(8);      // fp32 bit pattern
    constexpr uint32_t OUT_SLOTS = get_compile_time_arg_val(9);
    constexpr uint32_t HIt = get_compile_time_arg_val(10);   // heads * tile rows, one design's bands
    // 1: write bf16 tiles (the fused-SDPA mask, sparse_mask_bf16), the bias as read and the
    // fill's top half; 0: the fp32 widen above.
    constexpr uint32_t OUT16 = get_compile_time_arg_val(11);
    constexpr auto bias_args = TensorAccessorArgs<12>();
    constexpr auto idx_args = TensorAccessorArgs<bias_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();
#ifdef CSS_CPT
    // Chunk-skipping SDPA mask (sdpa_generic css): row q of the list tensor names the key chunks of
    // CSS_CPT tiles that query chunk q (CSS_QCT tile rows) visits, and the SDPA reads no other tile.
    // An empty tile outside those chunks is never read, so it is not written.
    constexpr auto css_args = TensorAccessorArgs<out_args.next_compile_time_args_offset()>();
    const auto s_css = TensorAccessor(css_args, get_common_arg_val<uint32_t>(3), CSS_W * 4);
    volatile tt_l1_ptr uint32_t* css_l1 = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(CSS_CB));
    constexpr uint32_t NEED_WORDS = (Jt / CSS_CPT + 31) / 32;
    uint32_t need[NEED_WORDS];
    uint32_t css_q = 0xFFFFFFFFu;
#endif

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t IDX_ROW_WORDS = K;
    constexpr uint32_t BIAS_TILE_BYTES = 32 * 32 * 2;
    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * (OUT16 ? 2 : 4);
    constexpr uint32_t OUT_TILE_WORDS = 32 * 32;

    const auto s_bias = TensorAccessor(bias_args, bias_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);

    volatile tt_l1_ptr uint32_t* idx_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_idx));
    const uint32_t bias_base = get_write_ptr(cb_bias);
    volatile tt_l1_ptr uint16_t* bias_l1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(bias_base);
    const uint32_t out_base = get_write_ptr(cb_out);

    // Prime every output slot with the fill value once. Slot OUT_SLOTS is never written after this: the all-fill
    // tile every empty key tile is sent from, and the source every used slot is refilled from.
    for (uint32_t slot = 0; slot <= OUT_SLOTS; ++slot) {
        for (uint32_t w = 0; w < OUT_TILE_WORDS; ++w) {
            put<OUT16>(out_base + slot * OUT_TILE_BYTES, w, fill);
        }
    }

    constexpr uint32_t NO_TILE = 0xFFFFFFFFu;
    const uint64_t fill_noc = get_noc_addr(out_base + OUT_SLOTS * OUT_TILE_BYTES);
    uint32_t slot_jt[OUT_SLOTS];
    for (uint32_t slot = 0; slot < OUT_SLOTS; ++slot) {
        slot_jt[slot] = NO_TILE;
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
#ifdef CSS_CPT
        const uint32_t q = b * ((It + CSS_QCT - 1) / CSS_QCT) + it / CSS_QCT;
        const bool css_fresh = q != css_q;
        if (css_fresh) {
            noc_async_read(s_css.get_noc_addr(q), reinterpret_cast<uint32_t>(css_l1), CSS_W * 4);
        }
#endif
        noc_async_read_barrier();
#ifdef CSS_CPT
        if (css_fresh) {
            css_q = q;
            for (uint32_t w = 0; w < NEED_WORDS; ++w) {
                need[w] = 0;
            }
            for (uint32_t i = 0; i < css_l1[0]; ++i) {
                const uint32_t c = css_l1[1 + i];
                need[c >> 5] |= 1u << (c & 31);
            }
        }
#endif

        for (uint32_t r = 0; r < rows; ++r) {
            cur[r] = 0;
        }
        // Which key tiles of this band hold any index: one pass over rows x K instead of a cursor probe per row
        // per tile. Most tiles hold none (128 keys of ~6000), and those go out from the fill tile untouched,
        // with no scan, no slot and no refill. On Wormhole the probes, not the write, were the op's cost.
        constexpr uint32_t OCC_WORDS = (Jt + 31) / 32;
        uint32_t occ[OCC_WORDS];
        for (uint32_t w = 0; w < OCC_WORDS; ++w) {
            occ[w] = 0;
        }
        for (uint32_t r = 0; r < (MASKW_DIAG == 3 ? 0 : rows); ++r) {
            volatile tt_l1_ptr uint32_t* irow = idx_l1 + r * IDX_ROW_WORDS;
            for (uint32_t sidx = 0; sidx < K; ++sidx) {
                const uint32_t jt = irow[sidx] >> 5;
                if (jt < Jt) {
                    occ[jt >> 5] |= 1u << (jt & 31);
                }
            }
        }
        uint32_t used = 0;

        // Restore every used slot to all-fill from the fill tile. Called only with their writes flushed.
        auto refill = [&]() {
            for (uint32_t slot = 0; slot < OUT_SLOTS; ++slot) {
                if (slot_jt[slot] != NO_TILE) {
                    noc_async_read(fill_noc, out_base + slot * OUT_TILE_BYTES, OUT_TILE_BYTES);
                    slot_jt[slot] = NO_TILE;
                }
            }
            noc_async_read_barrier();
        };

        uint32_t out_page = group * Jt;
        for (uint32_t jt = 0; jt < Jt; ++jt) {
            if (MASKW_DIAG == 1 || MASKW_DIAG == 3 || !(occ[jt >> 5] & (1u << (jt & 31)))) {
#ifdef CSS_CPT
                const uint32_t kc = jt / CSS_CPT;
                if (!(need[kc >> 5] & (1u << (kc & 31)))) {
                    ++out_page;
                    continue;
                }
#endif
                out_write(out_base + OUT_SLOTS * OUT_TILE_BYTES, s_out, out_page, OUT_TILE_BYTES);
                ++out_page;
                continue;
            }
            const uint32_t slot = used++ & (OUT_SLOTS - 1);
            const uint32_t out_addr_l1 = out_base + slot * OUT_TILE_BYTES;

            const uint32_t col_lo = jt * 32;
            const uint32_t col_hi = col_lo + 32;
            for (uint32_t r = 0; r < rows; ++r) {
                volatile tt_l1_ptr uint32_t* irow = idx_l1 + r * IDX_ROW_WORDS;
                uint32_t sidx = cur[r];
                while (sidx < K) {
                    const uint32_t j = irow[sidx];
                    if (j >= col_hi) {
                        break;
                    }
                    const uint32_t bv =
                        bias_l1[(sidx >> 5) * (BIAS_TILE_BYTES / 2) + face_off(r, sidx & 31)];
                    put<OUT16>(out_addr_l1, face_off(r, j - col_lo), bv << 16);
                    ++sidx;
                }
                cur[r] = sidx;
            }
            slot_jt[slot] = jt;

            out_write(out_addr_l1, s_out, out_page, OUT_TILE_BYTES);
            ++out_page;
            // Every slot is written out after this, so all of them can be refilled for jt + 1.
            if (slot == OUT_SLOTS - 1) {
                noc_async_write_barrier();
                refill();
            }
        }
        // Jt is not in general a multiple of OUT_SLOTS. Drain, then hand the next band a set of clean slots.
        noc_async_write_barrier();
        refill();
    }
}
