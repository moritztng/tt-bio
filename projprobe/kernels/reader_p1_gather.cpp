// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 1(ii) -- the exact-trilinear gather. Eight corners per (orientation, pixel) pair, one 8 B
// NoC read each, straight into eight dense [orientation, pixel] slot tiles.
//
// The addresses come from the compute unit, not from here (§4.4): if the reader recomputed
// floor(xp) itself, one ulp of disagreement would fetch a different cell and blend it with the right
// weights. Phase 1(i) emits float(voxel_index) + 2^23, so the mantissa IS the index and recovering it
// is one AND. Outside the radius it emits the sentinel 2^23-1, which this kernel tests and skips --
// §2.4b measured that at 22.3% of all pairs.
//
// §4.4 says "byte address" but its own bound (addr < 2^23 against nvox = 3,960,100) only holds for
// the VOXEL index; the byte address is 8x larger and does not fit in a float mantissa. So the index
// is what crosses, and the shift left by 3 happens here.
//
// The addressing is trivial because of the paired-column layout (§8.7 correction): each pixel owns
// two adjacent columns, re and im. Within a tile's stored face order consecutive columns are
// consecutive words, and a pair's two columns never straddle a face boundary, so linear offsets 2k
// and 2k+1 are exactly one pair. This kernel walks k and never does face arithmetic -- and the same
// linear offset indexes the slot tiles, because they have the same layout.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_addr = get_compile_time_arg_val(0);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(1);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(3);
    constexpr uint32_t mdlX = get_compile_time_arg_val(4);
    constexpr uint32_t mdlXY = get_compile_time_arg_val(5);
    constexpr uint32_t cb_mdl = get_compile_time_arg_val(6);
    constexpr uint32_t mdl_pages = get_compile_time_arg_val(7);
    constexpr auto a_args = TensorAccessorArgs<8>();
    constexpr auto m_args = TensorAccessorArgs<a_args.next_compile_time_args_offset()>();

    const uint32_t a_addr = get_arg_val<uint32_t>(0);
    const uint32_t m_addr = get_arg_val<uint32_t>(1);

    const auto sa = TensorAccessor(a_args, a_addr, tile_bytes);
    const auto sm = TensorAccessor(m_args, m_addr, tile_bytes);

    // Stage the model into this core's own L1 before gathering. §4.1's production kernel keeps it
    // L1-resident and sharded across the grid for exactly this reason: the gather needs byte-granular
    // random access, and an interleaved DRAM or L1 tensor gives page-granular access spread over
    // banks, which is not addressable by raw offset.
    cb_reserve_back(cb_mdl, mdl_pages);
    const uint32_t mdl_l1 = get_write_ptr(cb_mdl);
    for (uint32_t i = 0; i < mdl_pages; ++i) {
        noc_async_read_page(i, sm, mdl_l1 + i * tile_bytes);
    }
    noc_async_read_barrier();
    cb_push_back(cb_mdl, mdl_pages);

    // The eight trilinear corners, in the slot order the blend expects:
    // 000 100 010 110 001 101 011 111, i.e. bit 0 = x, bit 1 = y, bit 2 = z.
    const uint32_t off[8] = {0,         1,
                             mdlX,      mdlX + 1,
                             mdlXY,     mdlXY + 1,
                             mdlXY + mdlX, mdlXY + mdlX + 1};

    constexpr uint32_t PAIRS = 512;              // 32 rows x 16 pixels, two columns each
    constexpr uint32_t SENTINEL = 0x7FFFFFu;

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_reserve_back(cb_addr, 1);
        noc_async_read_page(b, sa, get_write_ptr(cb_addr));
        noc_async_read_barrier();
        cb_push_back(cb_addr, 1);

        cb_wait_front(cb_addr, 1);
        const uint32_t ap = get_read_ptr(cb_addr);
        cb_reserve_back(cb_slot, 8);
        const uint32_t sp = get_write_ptr(cb_slot);

        // Skipped pairs must read as zero rather than as whatever the CB held last time round.
        // The assembled kernel does not need this -- the radius mask multiplies the blend result --
        // but leaving stale data here would make a grading failure unattributable.
        volatile tt_l1_ptr uint32_t *sz = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sp);
        for (uint32_t i = 0; i < 8 * (tile_bytes / 4); ++i) {
            sz[i] = 0;
        }

        volatile tt_l1_ptr uint32_t *av = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(ap);
        for (uint32_t k = 0; k < PAIRS; ++k) {
            const uint32_t idx = av[2 * k] & SENTINEL;     // the mantissa is the voxel index
            if (idx == SENTINEL) {
                continue;                                  // outside the radius: leave it zero
            }
            const uint32_t dst_off = 2 * k * 4;            // same linear offset in every slot tile
            for (uint32_t s = 0; s < 8; ++s) {
                // The production kernel reads another core's L1 shard (§4.1); this arm reads its
                // own, which is the same NoC path with the same issue cost, and lets the screen run
                // against a model small enough to be resident on one core.
                const uint64_t src = get_noc_addr(mdl_l1 + ((idx + off[s]) << 3));
                noc_async_read(src, sp + s * tile_bytes + dst_off, 8);
            }
        }
        noc_async_read_barrier();
        cb_push_back(cb_slot, 8);
        cb_pop_front(cb_addr, 1);
    }
}
