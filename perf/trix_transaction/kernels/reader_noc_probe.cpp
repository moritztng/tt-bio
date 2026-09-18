// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOC transaction counter probe.
//
// Replays the page-read pattern of `reader_reblock_permute.cpp` against the same DRAM tensor and
// reports, per core, the NIU master counters the hardware keeps: read requests SENT, 32-byte data
// words RECEIVED, read responses RECEIVED, and the peak outstanding-request depth. Those are
// registers, not a model: bytes per transaction is word_delta * 32 / req_delta and needs no roof.
//
// It does not compute anything and never drains a circular buffer -- c_0 is a plain L1 scratch
// window, addressed directly, so the reader runs at whatever rate the memory system gives it with
// no consumer in the way. That isolates the read side of the channel move from its compute and its
// writer, which is the whole point: the shipped op's rate mixes all three.
//
// Arms are runtime args, so one compiled program serves the whole ladder:
//   `rows`        tiles read per (group, channel-tile), 32 in production
//   `batch`       reads issued before a barrier; 1 is production
//   `bank_step`   added to the page once per row, ON TOP of the natural row stride, to move the
//                 read off the bank the natural stride pins it to. 0 is production. The pages it
//                 produces are still inside the tensor (the host sizes the walk so they are), the
//                 VALUES are wrong, and this arm is a timing/counter arm only, never a correctness
//                 claim.
//   `ct_inner`    1 walks the channel tile in the inner loop instead of the row, which is the
//                 access order a restructured kernel would have. Production is 0.
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t dst_addr = get_common_arg_val<uint32_t>(1);

    uint32_t a = 0;
    const uint32_t first_group = get_arg_val<uint32_t>(a++);
    const uint32_t num_groups = get_arg_val<uint32_t>(a++);
    const uint32_t Nt = get_arg_val<uint32_t>(a++);
    const uint32_t Ct = get_arg_val<uint32_t>(a++);
    const uint32_t rows = get_arg_val<uint32_t>(a++);
    const uint32_t batch = get_arg_val<uint32_t>(a++);
    const uint32_t bank_step = get_arg_val<uint32_t>(a++);
    const uint32_t ct_inner = get_arg_val<uint32_t>(a++);
    const uint32_t sample = get_arg_val<uint32_t>(a++);
    const uint32_t core_id = get_arg_val<uint32_t>(a++);

    constexpr uint32_t cb_scratch = 0;
    constexpr uint32_t TILE_BYTES = 2048;
    constexpr uint32_t SLOTS = 8;

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto s = TensorAccessor(src_args, src_addr);
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();
    const auto d = TensorAccessor(dst_args, dst_addr);

    const uint32_t base = get_write_ptr(cb_scratch);
    const uint32_t row_stride = Nt * Ct;

    const uint32_t req0 = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_REQ_SENT);
    const uint32_t word0 = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_DATA_WORD_RECEIVED);
    const uint32_t resp0 = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
    uint32_t nreads = 0;
    uint32_t maxout = 0;
    uint32_t slot = 0;
    uint32_t pending = 0;

    for (uint32_t gi = 0; gi < num_groups; ++gi) {
        const uint32_t group = first_group + gi;
        const uint32_t it = group / Nt;
        const uint32_t jt = group % Nt;
        const uint32_t first_page = (it * 32 * Nt + jt) * Ct;

        if (ct_inner == 0) {
            for (uint32_t ct = 0; ct < Ct; ++ct) {
                uint32_t page = first_page + ct;
                for (uint32_t il = 0; il < rows; ++il) {
                    noc_async_read_page(page, s, base + slot * TILE_BYTES);
                    slot = (slot + 1) & (SLOTS - 1);
                    ++nreads;
                    page += row_stride + bank_step;
                    if (++pending >= batch) {
                        if (sample) {
                            const uint32_t o =
                                NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_REQ_SENT) -
                                NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
                            if (o > maxout) { maxout = o; }
                        }
                        noc_async_read_barrier();
                        pending = 0;
                    }
                }
            }
        } else {
            for (uint32_t il = 0; il < rows; ++il) {
                const uint32_t row_page = first_page + il * row_stride;
                for (uint32_t ct = 0; ct < Ct; ++ct) {
                    noc_async_read_page(row_page + ct, s, base + slot * TILE_BYTES);
                    slot = (slot + 1) & (SLOTS - 1);
                    ++nreads;
                    if (++pending >= batch) {
                        if (sample) {
                            const uint32_t o =
                                NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_REQ_SENT) -
                                NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);
                            if (o > maxout) { maxout = o; }
                        }
                        noc_async_read_barrier();
                        pending = 0;
                    }
                }
            }
        }
    }
    if (pending) {
        noc_async_read_barrier();
        pending = 0;
    }

    const uint32_t req1 = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_REQ_SENT);
    const uint32_t word1 = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_DATA_WORD_RECEIVED);
    const uint32_t resp1 = NOC_STATUS_READ_REG(noc_index, NIU_MST_RD_RESP_RECEIVED);

    // The report goes out AFTER the last snapshot, so the write never appears in the read counters.
    const uint32_t rep = base + SLOTS * TILE_BYTES;
    volatile tt_l1_ptr uint32_t* o = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(rep);
    o[0] = req1 - req0;
    o[1] = word1 - word0;
    o[2] = resp1 - resp0;
    o[3] = nreads;
    o[4] = maxout;
    o[5] = num_groups;
    o[6] = core_id;
    o[7] = row_stride;
    noc_async_write(rep, d.get_noc_addr(core_id), 32);
    noc_async_write_barrier();
}
