// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// reblock_permute_back reader (multi-core). The INVERSE channel move:
// permute(x, (0,2,3,1)) for x [1, C, N, N] -> [1, N, N, C], C a multiple of 32,
// Ct = C/32 channel tiles, N a multiple of 32 (the ragged case is refused by the
// host gate, so nothing here is padded).
//
// A group is (it, jt, ct), flattened as g = (it*Nt + jt)*Ct + ct, and owns the 32
// output tiles of the 32 rows i = it*32 + il. It reads the 32 input tiles
//   { (ct*32 + cl, it, jt) : cl in [0,32) },  page = (ct*32+cl)*Nt*Nt + it*Nt + jt
// and builds, for each il, the gathered tile
//   T_il[cl, jl] = in_tile_cl[il, jl] = x[ct*32+cl, it*32+il, jt*32+jl].
// The compute kernel transposes it, giving the real output tile
//   out[jl, cl] = T_il[cl, jl] = y[it*32+il, jt*32+jl, ct*32+cl].
//
// Why the group reads whole tiles first: the 32 sources of one output tile are 32
// DIFFERENT DRAM pages and only one 32-element row of each is wanted. Reading
// those rows straight from DRAM is a 2-byte-strided sub-line read, which
// Blackhole quantizes to 16 B blocks. So the group takes 32 aligned 2 KB reads
// into a contiguous L1 scratch window and the gather is L1 -> L1, where any
// 16 B-aligned length is exact. This is the mirror of the forward direction,
// which pays the same cost on its writer.
//
// THE BINDING RESOURCE IS INSTRUCTION COUNT ON THIS RISC, not bytes: the gather
// is 64 transactions per output tile whatever the kernel structure, so the only
// axis left is instructions per transaction. Two things are done for it, both
// copied from the forward writer: the reads issue
// `noc_async_read_one_packet_with_state` with the NOC coordinates and the 32-byte
// length set ONCE per invocation, and every address in the gather body is an
// induction variable with the loop split at the face boundary, so no multiply,
// divide or branch survives inside it.
//
// Per-core CB accounting: num_groups * 32 pushes to c_0, matched by compute.
// A core with num_groups == 0 reads and pushes nothing and exits cleanly.
//
// IMPORTANT: use the api/-prefixed include path (bare dataflow_api.h is a known
// device-wedge cause in this codebase).
#include "api/dataflow/dataflow_api.h"

void kernel_main() {
    // src_addr is the only value that changes between calls at a fixed (N, C, buffer type, grid),
    // so it lives in the common runtime args and the host caches the whole ProgramDescriptor.
    const uint32_t src_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t start_group = get_arg_val<uint32_t>(0);
    const uint32_t num_groups = get_arg_val<uint32_t>(1);
    const uint32_t Nt = get_arg_val<uint32_t>(2);
    const uint32_t Ct = get_arg_val<uint32_t>(3);

    constexpr uint32_t element_size = get_compile_time_arg_val(0);
    constexpr uint32_t scratch_cb_id = get_compile_time_arg_val(1);  // c_24
    constexpr uint32_t in_cb_id = get_compile_time_arg_val(2);       // c_0, gathered tiles -> compute
    constexpr uint32_t TILE_HEIGHT = get_compile_time_arg_val(3);    // 32
    constexpr uint32_t TILE_WIDTH = get_compile_time_arg_val(4);     // 32
    constexpr uint32_t FACE_HEIGHT = get_compile_time_arg_val(5);    // 16
    constexpr uint32_t FACE_WIDTH = get_compile_time_arg_val(6);     // 16
    constexpr auto src_args = TensorAccessorArgs<7>();

    constexpr uint32_t NUM_FACES_W = TILE_WIDTH / FACE_WIDTH;                 // 2
    constexpr uint32_t face_height_width = FACE_HEIGHT * FACE_WIDTH;          // 256
    constexpr uint32_t tile_bytes = TILE_HEIGHT * TILE_WIDTH * element_size;  // 2048
    constexpr uint32_t FACE_ROW_BYTES = FACE_WIDTH * element_size;            // 32
    constexpr uint32_t FACE_BYTES = face_height_width * element_size;         // 512

    const auto s = TensorAccessor(src_args, src_addr);

    const uint32_t NtNt = Nt * Nt;
    const uint32_t NtCt = Nt * Ct;
    const uint32_t end_group = start_group + num_groups;

    // The scratch window is this RISC's private staging area, not a producer/consumer queue: the
    // same kernel writes it and reads it, so it is reserved ONCE and addressed directly, the way the
    // forward direction's writer holds its own staging CB.
    cb_reserve_back(scratch_cb_id, TILE_HEIGHT);
    const uint32_t group_l1_base = get_write_ptr(scratch_cb_id);

    // Every gather read is local L1 -> L1 at exactly FACE_ROW_BYTES, so the coordinates and the
    // length are the same for all of them and go into the read command buffer once per GROUP rather
    // than once per transaction.
    //
    // Two things about this line were each worth a wrong kernel, and both are silent:
    //
    //   * it is uint64_t, not uint32_t. `get_noc_addr` returns a 64-bit NOC address whose high bits
    //     are the target core's x/y; truncating it to 32 bits keeps only the local offset, so the
    //     state names core (0,0) and every gather read fetches another core's L1. Measured
    //     `torch.equal` false at every shape at a uniform 5 GB/s -- SLOWER than the wrong-state case
    //     below, because the reads become real NOC hops.
    //   * it is re-issued per group and not once per kernel. This kernel also calls
    //     `noc_async_read_page` for the group's DRAM tiles and that shares the same read command
    //     buffer, so the DRAM reads overwrite the one-packet state. Setting it once at entry
    //     measured false everywhere at 9 GB/s; setting it once lazily after the first group's reads
    //     measured TRUE at exactly the shapes where no core owns two groups (N <= 320 at C = 32) and
    //     false beyond, which is the shape-dependent failure that makes this trap dangerous. The
    //     forward writer gets to set it once because a writer issues no reads but its own gather.
    const uint64_t gather_state_addr = get_noc_addr(group_l1_base);

    for (uint32_t group = start_group; group < end_group; ++group) {
        const uint32_t it = group / NtCt;
        const uint32_t rem = group - it * NtCt;
        const uint32_t jt = rem / Ct;
        const uint32_t ct = rem - jt * Ct;

        // 1) the group's 32 input tiles, as 32 aligned 2 KB DRAM reads into contiguous L1
        {
            uint32_t page = (ct * TILE_HEIGHT) * NtNt + it * Nt + jt;
            uint32_t dst = group_l1_base;
            for (uint32_t cl = 0; cl < TILE_HEIGHT; ++cl) {
                noc_async_read_page(page, s, dst);
                page += NtNt;
                dst += tile_bytes;
            }
        }
        noc_async_read_barrier();
        noc_async_read_one_packet_set_state(gather_state_addr, FACE_ROW_BYTES);

        // 2) T_il[cl, jl] = in_tile_cl[il, jl]: row `il` of every source tile becomes row `cl` of
        //    the gathered tile. Rows [0,16) of the destination live in faces 0 and 1, rows [16,32)
        //    in faces 2 and 3, so the loop is split there and the body is pure induction.
        for (uint32_t il = 0; il < TILE_HEIGHT; ++il) {
            cb_reserve_back(in_cb_id, 1);
            const uint32_t dst_tile_base = get_write_ptr(in_cb_id);

            // Source offset of row `il`, invariant in `cl`: (il / 16) picks the source face pair,
            // (il % 16) the row inside it.
            const uint32_t src_il = group_l1_base
                                  + (il / FACE_HEIGHT) * NUM_FACES_W * FACE_BYTES
                                  + (il % FACE_HEIGHT) * FACE_ROW_BYTES;
            uint32_t s0 = src_il;                 // face_w = 0
            uint32_t s1 = src_il + FACE_BYTES;    // face_w = 1
            uint32_t d0 = dst_tile_base;
            uint32_t d1 = dst_tile_base + FACE_BYTES;
            for (uint32_t cl = 0; cl < FACE_HEIGHT; ++cl) {
                noc_async_read_one_packet_with_state(s0, d0);
                noc_async_read_one_packet_with_state(s1, d1);
                s0 += tile_bytes;
                s1 += tile_bytes;
                d0 += FACE_ROW_BYTES;
                d1 += FACE_ROW_BYTES;
            }
            d0 = dst_tile_base + 2 * FACE_BYTES;
            d1 = dst_tile_base + 3 * FACE_BYTES;
            for (uint32_t cl = FACE_HEIGHT; cl < TILE_HEIGHT; ++cl) {
                noc_async_read_one_packet_with_state(s0, d0);
                noc_async_read_one_packet_with_state(s1, d1);
                s0 += tile_bytes;
                s1 += tile_bytes;
                d0 += FACE_ROW_BYTES;
                d1 += FACE_ROW_BYTES;
            }
            noc_async_read_barrier();  // drain the L1 -> L1 gather before handing the tile to compute
            cb_push_back(in_cb_id, 1);
        }

    }
}
