// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Phase 3 reader -- feeds the coordinate stage AND performs the gather, because the inverted
// dataflow (§4.4) means those two are the same kernel's job: it hands the compute unit its euler and
// lattice tiles, waits for the address tile the compute unit produces from them, gathers the eight
// corners, and hands those back.
//
// Alignment rules from §8.9 apply throughout: a NoC transfer needs 16 B at both ends, a misaligned
// source silently returns the neighbouring block and a misaligned destination hangs the device. Each
// x-pair therefore reads the two aligned 16 B blocks that can contain (x0, x0+1) into a CB-backed
// scratch, and the wanted 8 B are copied out with plain pointer writes, which never touch the NoC.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    constexpr uint32_t cb_e = get_compile_time_arg_val(0);
    constexpr uint32_t cb_xy = get_compile_time_arg_val(1);
    constexpr uint32_t cb_c = get_compile_time_arg_val(2);
    constexpr uint32_t cb_addr = get_compile_time_arg_val(3);
    constexpr uint32_t cb_slot = get_compile_time_arg_val(4);
    constexpr uint32_t cb_mdl = get_compile_time_arg_val(5);
    constexpr uint32_t cb_scr = get_compile_time_arg_val(6);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t n_blocks = get_compile_time_arg_val(8);
    constexpr uint32_t mdlX = get_compile_time_arg_val(9);
    constexpr uint32_t mdlXY = get_compile_time_arg_val(10);
    constexpr uint32_t mdl_pages = get_compile_time_arg_val(11);
    // Ablation selector, for §8.15's cost attribution. Bit 0 drops the per-block zero-fill, bit 1
    // drops the NoC reads, bit 2 drops the parity-select pointer copies. Bit 3 issues ONE aligned
    // 16 B read per x-pair instead of two, which is what an x-parity model relayout (§4.7 / §10
    // item 7) would buy: with x0 always even the pair (x0, x0+1) sits inside a single aligned block.
    // Any nonzero value makes the kernel WRONG on purpose; these arms measure where the time goes.
    constexpr uint32_t abl = get_compile_time_arg_val(12);
    constexpr auto e_args = TensorAccessorArgs<13>();
    constexpr auto x_args = TensorAccessorArgs<e_args.next_compile_time_args_offset()>();
    constexpr auto c_args = TensorAccessorArgs<x_args.next_compile_time_args_offset()>();
    constexpr auto m_args = TensorAccessorArgs<c_args.next_compile_time_args_offset()>();

    const auto se = TensorAccessor(e_args, get_arg_val<uint32_t>(0), tile_bytes);
    const auto sx = TensorAccessor(x_args, get_arg_val<uint32_t>(1), tile_bytes);
    const auto sc_t = TensorAccessor(c_args, get_arg_val<uint32_t>(2), tile_bytes);
    const auto sm = TensorAccessor(m_args, get_arg_val<uint32_t>(3), tile_bytes);

    cb_reserve_back(cb_mdl, mdl_pages);
    const uint32_t mdl_l1 = get_write_ptr(cb_mdl);
    for (uint32_t i = 0; i < mdl_pages; ++i) {
        noc_async_read_page(i, sm, mdl_l1 + i * tile_bytes);
    }
    cb_reserve_back(cb_c, 9);
    uint32_t p = get_write_ptr(cb_c);
    for (uint32_t i = 0; i < 9; ++i) {
        noc_async_read_page(i, sc_t, p);
        p += tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_mdl, mdl_pages);
    cb_push_back(cb_c, 9);

    cb_reserve_back(cb_scr, 1);
    const uint32_t scr = get_write_ptr(cb_scr);
    volatile tt_l1_ptr uint32_t *sc = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(scr);

    const uint32_t xoff[4] = {0, mdlX, mdlXY, mdlXY + mdlX};
    constexpr uint32_t PAIRS = 512;
    constexpr uint32_t SENTINEL = 0x7FFFFFu;

    for (uint32_t b = 0; b < n_blocks; ++b) {
        cb_reserve_back(cb_e, 6);
        p = get_write_ptr(cb_e);
        for (uint32_t i = 0; i < 6; ++i) {
            noc_async_read_page(b * 6 + i, se, p);
            p += tile_bytes;
        }
        cb_reserve_back(cb_xy, 2);
        p = get_write_ptr(cb_xy);
        for (uint32_t i = 0; i < 2; ++i) {
            noc_async_read_page(b * 2 + i, sx, p);
            p += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_e, 6);
        cb_push_back(cb_xy, 2);

        // The compute unit turns those into an address tile. Wait for it, then gather.
        cb_wait_front(cb_addr, 1);
        const uint32_t ap = get_read_ptr(cb_addr);
        cb_reserve_back(cb_slot, 8);
        const uint32_t sp = get_write_ptr(cb_slot);

        // The zero-fill is hoisted: it only has to run while the CB slots are still uninitialised
        // L1, because after that any stale value a skipped pair leaves behind is a finite number from
        // an earlier block and the radius mask multiplies it out. Running it every block cost
        // 35.5 ns/pair (§8.15) for nothing.
        if constexpr ((abl & 1u) == 0u) {
            if (b < 2) {
                volatile tt_l1_ptr uint32_t *sz = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(sp);
                for (uint32_t i = 0; i < 8 * (tile_bytes / 4); ++i) {
                    sz[i] = 0;
                }
            }
        }

        volatile tt_l1_ptr uint32_t *av = reinterpret_cast<volatile tt_l1_ptr uint32_t *>(ap);
        for (uint32_t k = 0; k < PAIRS; ++k) {
            const uint32_t idx = av[2 * k] & SENTINEL;
            if (idx == SENTINEL) {
                continue;
            }
            uint32_t base[4];
            for (uint32_t j = 0; j < 4; ++j) {
                const uint32_t v0 = idx + xoff[j];
                const uint32_t a = v0 & ~1u;
                base[j] = (v0 - a) * 2;
                if constexpr ((abl & 2u) == 0u) {
                    noc_async_read(get_noc_addr(mdl_l1 + (a << 3)), scr + j * 32, 16);
                    if constexpr ((abl & 8u) == 0u) {
                        noc_async_read(get_noc_addr(mdl_l1 + ((a + 2) << 3)), scr + j * 32 + 16, 16);
                    }
                }
            }
            noc_async_read_barrier();
            const uint32_t dst_w = 2 * k;
            for (uint32_t j = 0; j < 4 && ((abl & 4u) == 0u); ++j) {
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
