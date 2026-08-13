// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// RFD3 fused attention scores, L6c: the SAME work as reader_fused_scores.cpp + writer_fused_scores
// .cpp, but with the bias build shared across BOTH data-movement RISCs instead of sitting on one.
//
// WHY. The fused op measures 1.673 ms/call at [1,4,3359,3360] while its own DRAM traffic is 0.70 ms
// at this card's measured 385 GB/s roof, and the config sweep says the pipeline is not the
// constraint (doubling the scores depth or the write window changes nothing; only starving the bias
// ring hurts). What is left is the poke walk itself -- 1.72 M pokes, as many repair writes, and the
// index scan -- all of it volatile L1 traffic on ONE RISC. L6a measured the same work halving on
// two RISCs, 1.652 -> 0.932, because the (head, tile-row) bands are independent and need no
// semaphore. This kernel is that split, with the output drain folded into one of the two.
//
// ONE SOURCE, TWO ROLES. Both RISCs run this file. Compile-time PHASE picks which bands a RISC
// owns: local band b belongs to PHASE ((b + phase_offset) & 1), and phase_offset is per core so
// that a core with an odd band count gives its extra band to a different RISC than its neighbour
// does -- without it the even-phase RISC would carry 260 of the 420 bands instead of 210.
// Compile-time DRAIN says which RISC also writes cb_out to DRAM.
//
// THE DEADLOCK, AND THE ONE LINE THAT AVOIDS IT. The draining RISC has two jobs whose order is not
// fixed: it must produce its own bias tiles AND drain the compute kernel's output. If it blocks in
// cb_reserve_back on its bias ring while the compute kernel is blocked on cb_out being full,
// neither can move and the core hangs. So every wait on the produce side goes through
// `reserve_or_pump`, which drains whatever output happens to be available instead of sleeping. That
// is enough for progress: the compute kernel can always advance if its inputs have data or its
// output has room, and pumping is exactly what creates the room.
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
    const uint32_t out_addr = get_common_arg_val<uint32_t>(3);

    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t phase_offset = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scores = get_compile_time_arg_val(0);
    constexpr uint32_t cb_bias = get_compile_time_arg_val(1);
    constexpr uint32_t cb_idx = get_compile_time_arg_val(2);   // RISC-local scratch
    constexpr uint32_t cb_pb = get_compile_time_arg_val(3);    // RISC-local scratch
    constexpr uint32_t cb_cur = get_compile_time_arg_val(4);   // RISC-local scratch
    constexpr uint32_t cb_out = get_compile_time_arg_val(5);
    constexpr uint32_t It = get_compile_time_arg_val(6);
    constexpr uint32_t Jt = get_compile_time_arg_val(7);
    constexpr uint32_t Kt = get_compile_time_arg_val(8);
    constexpr uint32_t K = get_compile_time_arg_val(9);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t fill = get_compile_time_arg_val(11);    // fp32 bit pattern
    constexpr uint32_t SLOTS = get_compile_time_arg_val(12);   // cb_bias depth, power of two
    constexpr uint32_t PHASE = get_compile_time_arg_val(13);
    constexpr uint32_t DRAIN = get_compile_time_arg_val(14);
    constexpr uint32_t WINDOW = get_compile_time_arg_val(15);
    constexpr uint32_t OUT_SLOTS = get_compile_time_arg_val(16);  // cb_out depth, power of two
    // Diagnostic only, never set in production: skip the poke walk and the repair replay, keeping
    // every read, every CB handshake, the compute pass and the write. It answers "is this op bound
    // by the poke work at all", which the two-RISC split (L6c) implicitly bet on and lost.
    constexpr uint32_t NOPOKE = get_compile_time_arg_val(17);
    constexpr auto scores_args = TensorAccessorArgs<18>();
    constexpr auto pb_args = TensorAccessorArgs<scores_args.next_compile_time_args_offset()>();
    constexpr auto idx_args = TensorAccessorArgs<pb_args.next_compile_time_args_offset()>();
    constexpr auto out_args = TensorAccessorArgs<idx_args.next_compile_time_args_offset()>();

    constexpr uint32_t TILE_H = 32;
    constexpr uint32_t SCORES_TILE_BYTES = 32 * 32 * 2;
    constexpr uint32_t BIAS_TILE_BYTES = 32 * 32 * 2;
    constexpr uint32_t OUT_TILE_BYTES = 32 * 32 * 4;
    constexpr uint32_t OUT_TILE_WORDS = 32 * 32;

    const auto s_scores = TensorAccessor(scores_args, scores_addr);
    const auto s_pb = TensorAccessor(pb_args, pb_addr);
    const auto s_idx = TensorAccessor(idx_args, idx_addr);
    const auto s_out = TensorAccessor(out_args, out_addr);

    volatile tt_l1_ptr uint32_t* idx_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_idx));
    const uint32_t pb_base = get_write_ptr(cb_pb);
    volatile tt_l1_ptr uint16_t* pb_l1 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pb_base);
    volatile tt_l1_ptr uint32_t* slot_l1 =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_write_ptr(cb_cur));
    const uint32_t bias_base = get_write_ptr(cb_bias);
    const uint32_t out_base = get_read_ptr(cb_out);

    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias_base + slot * OUT_TILE_BYTES);
        for (uint32_t w = 0; w < OUT_TILE_WORDS; ++w) {
            p[w] = fill;
        }
    }

    // --- the drain side, only compiled in on one of the two RISCs -----------------------------
    // It walks EVERY band of the core in the order the compute kernel emits them, not just this
    // RISC's own bands: cb_out carries both phases interleaved.
    uint32_t d_left = DRAIN ? num_groups * Jt : 0;
    uint32_t d_page = start_group * Jt;
    uint32_t d_jt = 0;
    uint32_t d_popped = 0;

    auto drain_pump = [&]() {
        if constexpr (DRAIN) {
            if (d_left == 0) {
                return;
            }
            uint32_t n = Jt - d_jt;
            if (n > WINDOW) {
                n = WINDOW;
            }
            while (n > 0 && !cb_pages_available_at_front(cb_out, n)) {
                --n;
            }
            if (n == 0) {
                return;
            }
            for (uint32_t w = 0; w < n; ++w) {
                noc_async_write(out_base + ((d_popped + w) & (OUT_SLOTS - 1)) * OUT_TILE_BYTES,
                                s_out.get_noc_addr(d_page + w), OUT_TILE_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, n);
            d_popped += n;
            d_page += n;
            d_left -= n;
            d_jt += n;
            if (d_jt == Jt) {
                d_jt = 0;
            }
        }
    };

    // A wait that does useful work instead of spinning. On the non-draining RISC this is exactly
    // cb_reserve_back.
    auto reserve_or_pump = [&](uint32_t cb, uint32_t n) {
        if constexpr (DRAIN) {
            while (!cb_pages_reservable_at_back(cb, n)) {
                drain_pump();
            }
        }
        cb_reserve_back(cb, n);
    };

    constexpr uint32_t NO_TILE = 0xFFFFFFFFu;
    uint32_t slot_jt[SLOTS];
    uint32_t slot_rows[SLOTS];
    for (uint32_t slot = 0; slot < SLOTS; ++slot) {
        slot_jt[slot] = NO_TILE;
        slot_rows[slot] = 0;
    }
    uint32_t cur[TILE_H];
    uint32_t pushed = 0;

    auto repair = [&](uint32_t slot) {
        if (NOPOKE || slot_jt[slot] == NO_TILE) {
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

    for (uint32_t b = 0; b < num_groups; ++b) {
        if (((b + phase_offset) & 1) != PHASE) {
            continue;
        }
        const uint32_t group = start_group + b;
        // h * It + it == group, so the page bases are group-major and need no divide.
        const uint32_t it = group % It;
        const uint32_t row_base = it * TILE_H;
        const uint32_t rows = (row_base + TILE_H <= L) ? TILE_H : (L - row_base);

        for (uint32_t r = 0; r < rows; ++r) {
            noc_async_read(s_idx.get_noc_addr(row_base + r),
                           reinterpret_cast<uint32_t>(idx_l1) + r * K * 4, K * 4);
        }
        for (uint32_t kt = 0; kt < Kt; ++kt) {
            noc_async_read(s_pb.get_noc_addr(group * Kt + kt), pb_base + kt * BIAS_TILE_BYTES,
                           BIAS_TILE_BYTES);
        }
        noc_async_read_barrier();

        for (uint32_t r = 0; r < rows; ++r) {
            cur[r] = 0;
        }

        for (uint32_t jt = 0; jt < Jt; ++jt) {
            reserve_or_pump(cb_scores, 1);
            noc_async_read(s_scores.get_noc_addr(group * Jt + jt), get_write_ptr(cb_scores),
                           SCORES_TILE_BYTES);

            reserve_or_pump(cb_bias, 1);
            const uint32_t slot = pushed & (SLOTS - 1);
            volatile tt_l1_ptr uint32_t* out_l1 =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(bias_base + slot * OUT_TILE_BYTES);
            repair(slot);

            volatile tt_l1_ptr uint32_t* sc = slot_l1 + slot * TILE_H;
            const uint32_t col_lo = jt * 32;
            const uint32_t col_hi = col_lo + 32;
            for (uint32_t r = 0; NOPOKE ? false : r < rows; ++r) {
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

        // Reclaim the ring before this band's index rows are overwritten: the replay is only
        // correct against the rows of the band that wrote them.
        reserve_or_pump(cb_bias, SLOTS);
        for (uint32_t slot = 0; slot < SLOTS; ++slot) {
            repair(slot);
        }
    }

    if constexpr (DRAIN) {
        while (d_left) {
            uint32_t n = Jt - d_jt;
            if (n > WINDOW) {
                n = WINDOW;
            }
            cb_wait_front(cb_out, n);
            for (uint32_t w = 0; w < n; ++w) {
                noc_async_write(out_base + ((d_popped + w) & (OUT_SLOTS - 1)) * OUT_TILE_BYTES,
                                s_out.get_noc_addr(d_page + w), OUT_TILE_BYTES);
            }
            noc_async_write_barrier();
            cb_pop_front(cb_out, n);
            d_popped += n;
            d_page += n;
            d_left -= n;
            d_jt += n;
            if (d_jt == Jt) {
                d_jt = 0;
            }
        }
    }
}
