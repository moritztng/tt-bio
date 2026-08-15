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
    constexpr uint32_t cb_scr = get_compile_time_arg_val(8);
    constexpr auto a_args = TensorAccessorArgs<9>();
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

    // The four x-pairs of the trilinear cell. Each supplies two corners, (x0) and (x0+1), which go to
    // slot tiles 2j and 2j+1.
    const uint32_t xoff[4] = {0, mdlX, mdlXY, mdlXY + mdlX};

    // A NoC transfer needs 16 B alignment at BOTH ends (§8.9). A misaligned SOURCE silently returns
    // the neighbouring aligned block; a misaligned DESTINATION hangs the device. A complex voxel is
    // 8 B, so an arbitrary corner is only 8 B aligned and cannot be fetched directly. Each x-pair
    // therefore reads the two aligned 16 B blocks that can contain (x0, x0+1), and the wanted 8 B are
    // copied out by plain pointer writes, which have no alignment constraint because they never touch
    // the NoC.
    //
    // The staging buffer is a CB, not a stack array. A stack array is 4 B aligned and lives wherever
    // the compiler puts it; using one hung the device. A CB write pointer is a real, tile-aligned L1
    // address, which is what a NoC destination has to be.
    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    volatile tt_l1_ptr uint32_t *sc = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(scr);

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_reserve_back(cb_addr, 1);
        noc_async_read_page(b, sa, get_write_ptr(cb_addr));
        noc_async_read_barrier();
        cb_push_back(cb_addr, 1);

        cb_wait_front(cb_addr, 1);
        const uint32_t ap = get_read_ptr(cb_addr);
        cb_reserve_back(cb_slot, 8);
        const uint32_t sp = get_write_ptr(cb_slot);

        volatile tt_l1_ptr uint32_t *sz = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sp);
        for (uint32_t i = 0; i < 8 * (tile_bytes / 4); ++i) {
            sz[i] = 0;
        }

        volatile tt_l1_ptr uint32_t *av = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(ap);

        for (uint32_t k = 0; k < PAIRS; ++k) {
            const uint32_t idx = av[2 * k] & SENTINEL;
            if (idx == SENTINEL) {
                continue;
            }
            // All eight reads for the pair are issued before a single barrier. Barriering per x-pair
            // instead serialises the gather and is the wrong shape for the assembled kernel.
            uint32_t base[4];
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t v0 = idx + xoff[j];
                const uint32_t a = v0 & ~1u;               // aligned block start, in voxels
                base[j] = (v0 - a) * 2;                    // 0 when x0 is even, 2 when odd
                noc_async_read(get_noc_addr(mdl_l1 + (a << 3)), scr + j * 32, 16);
                noc_async_read(get_noc_addr(mdl_l1 + ((a + 2) << 3)), scr + j * 32 + 16, 16);
            }
            noc_async_read_barrier();

            const uint32_t dst_w = 2 * k;                  // word offset, same in every slot tile
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t o = j * 8 + base[j];
                volatile tt_l1_ptr uint32_t *d0 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sp + (2 * j) * tile_bytes) + dst_w;
                volatile tt_l1_ptr uint32_t *d1 =
                    reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sp + (2 * j + 1) * tile_bytes) + dst_w;
                d0[0] = sc[o];
                d0[1] = sc[o + 1];
                d1[0] = sc[o + 2];
                d1[1] = sc[o + 3];
            }
        }
        cb_push_back(cb_slot, 8);
        cb_pop_front(cb_addr, 1);
    }
}
