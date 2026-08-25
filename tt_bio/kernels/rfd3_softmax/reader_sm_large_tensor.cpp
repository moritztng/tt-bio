// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"

// The wheel's `reader_unary_interleaved_sm_large_tensor.cpp`, verbatim, plus the value-tile stream
// that L5b needs, guarded by `PV_FUSED`. Diff it against the wheel copy: the guarded blocks are the
// whole of the change.
//
// The value tiles are read once per HEAD, not once per row. A core owns consecutive rows of one
// head almost always (760 rows over 110 cores, head boundaries every Ht=190 rows), so residency
// turns what would be a per-row re-read of Wt tiles -- as much DRAM traffic as the fused write it
// deletes -- into one load. The host gates on the CB budget that residency needs.

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t blk = get_arg_val<uint32_t>(1);
    const uint32_t pre_scale = get_arg_val<uint32_t>(2);
    // same arg index as in reader_unary and in reader_unary_transpose_wh_8bank
    const uint32_t NCht = get_arg_val<uint32_t>(3);
    const uint32_t tile_offset = get_arg_val<uint32_t>(4);
    const uint32_t Wt = get_arg_val<uint32_t>(5);
    uint32_t Ht = get_arg_val<uint32_t>(6);
    uint32_t mask_addr = get_arg_val<uint32_t>(7);
    uint32_t start_ht = get_arg_val<uint32_t>(8);
    uint32_t start_mask_id = get_arg_val<uint32_t>(9);
    const uint32_t reduce_scaler = get_arg_val<uint32_t>(10);
    uint32_t cb_length_t = get_arg_val<uint32_t>(11);
#ifdef PV_FUSED
    const uint32_t vv_addr = get_arg_val<uint32_t>(12);
    uint32_t head = get_arg_val<uint32_t>(13);
#endif
#if CAUSAL_MASK
    uint32_t mask_start_ht = get_arg_val<uint32_t>(12);
#endif

    constexpr auto src0_args = TensorAccessorArgs<0>();
    constexpr uint32_t cb_id_in0 = tt::CBIndex::c_0, cb_id_in1 = tt::CBIndex::c_1;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);

#if FUSED_SCALE_MASK
    constexpr auto mask_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();

    constexpr uint32_t cb_id_attn = 4;
    uint32_t mask_tile_bytes = get_tile_size(cb_id_attn);

    const auto addr_mask = TensorAccessor(mask_args, mask_addr, mask_tile_bytes);
    experimental::CircularBuffer cb_id_attn_obj(cb_id_attn);

#if CAUSAL_MASK
    constexpr uint32_t num_tiles_causal_mask = get_compile_time_arg_val(mask_args.next_compile_time_args_offset());

    uint32_t mask_ht = mask_start_ht;
#endif

    uint32_t ht = start_ht;
    uint32_t mask_id = start_mask_id;
    bool read_mask = true;
    constexpr auto cb_fused_scale = tt::CBIndex::c_3;
    generate_bcast_unary_scalar(cb_fused_scale, pre_scale);
#endif

    const auto src_a = TensorAccessor(src0_args, src_addr, src0_tile_bytes);

#ifdef PV_FUSED
    constexpr auto vv_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr uint32_t cb_id_vv = tt::CBIndex::c_1;
    const uint32_t vv_tile_bytes = get_tile_size(cb_id_vv);
    const auto vv_a = TensorAccessor(vv_args, vv_addr, vv_tile_bytes);
    experimental::CircularBuffer cb_id_vv_obj(cb_id_vv);
    uint32_t ht_pv = start_ht;
#endif

    {
        constexpr uint32_t cb_in_2 = tt::CBIndex::c_2;
        generate_reduce_scaler(cb_in_2, reduce_scaler);
    }

    experimental::Noc noc;
    experimental::CircularBuffer cb_id_in0_obj(cb_id_in0);

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
#if NUMERIC_STABLE
    // We need an extra pass to get numeric stable
    constexpr uint32_t total_passes = 3;
#else
    constexpr uint32_t total_passes = 2;
#endif
#if FUSED_SCALE_MASK
    uint32_t mask_id_offset = mask_id;
    uint32_t mask_index = mask_id;
#endif

    for (uint32_t ncht = 0; ncht < NCht; ncht++) {
#ifdef PV_FUSED
        // Load this head's value tiles before pushing any of the row's scores. The other order
        // deadlocks: compute waits on the value CB first, so it would never drain cb_in0, and the
        // reader would block filling it before it ever reached this read.
        if (ncht == 0 || ht_pv == 0) {
            cb_id_vv_obj.reserve_back(Wt);
            uint32_t vv_write_offset = 0;
            uint32_t vv_page = head * Wt;   // vv is [.., Kt, 1] tiles and Kt == Wt
            for (uint32_t k = 0; k < Wt; k++) {
                noc.async_read(
                    vv_a, cb_id_vv_obj, vv_tile_bytes, {.page_id = vv_page}, {.offset_bytes = vv_write_offset});
                vv_page++;
                vv_write_offset += vv_tile_bytes;
            }
            noc.async_read_barrier();
            cb_id_vv_obj.push_back(Wt);
        }
#endif
        // We need to pass once in order to calculate the sum and then to calculate the final value.
        for (uint32_t cur_pass = 0; cur_pass < total_passes; cur_pass++) {
            // We want to fill up the CB for input, and do so in chunks of blk
            uint32_t tile_index = tile_offset + (ncht * Wt);
#if FUSED_SCALE_MASK
            mask_index = mask_id_offset;
#endif
            for (uint32_t wt = 0; wt < Wt; wt += blk) {
                cb_id_in0_obj.reserve_back(blk);
                uint32_t write_offset = 0;
#if FUSED_SCALE_MASK
                cb_id_attn_obj.reserve_back(blk);
                uint32_t mask_write_offset = 0;
#endif
                for (uint32_t regs = 0; regs < blk; regs++) {
                    noc.async_read(
                        src_a, cb_id_in0_obj, src0_tile_bytes, {.page_id = tile_index}, {.offset_bytes = write_offset});
                    tile_index++;
                    write_offset += src0_tile_bytes;
#if FUSED_SCALE_MASK
                    noc.async_read(
                        addr_mask,
                        cb_id_attn_obj,
                        mask_tile_bytes,
                        {.page_id = mask_index},
                        {.offset_bytes = mask_write_offset});
                    mask_index++;
                    mask_write_offset += mask_tile_bytes;
#endif
                }
                noc.async_read_barrier();
                cb_id_in0_obj.push_back(blk);
#if FUSED_SCALE_MASK
                cb_id_attn_obj.push_back(blk);

#endif
            }
        }
#if CAUSAL_MASK
        ++ht;
        ++mask_ht;
        if (ht == Ht) {
            ht = 0;
            mask_ht = 0;
            mask_id_offset += num_tiles_causal_mask;
        } else if (mask_ht == Wt) {
            mask_ht = 0;
            mask_id = mask_id_offset;
        }
#elif FUSED_SCALE_MASK
        ht++;
        if (ht != Ht) {
            mask_index = mask_id_offset;
        } else {
            ht = 0;
            mask_id_offset = mask_index;
        }

#endif
#ifdef PV_FUSED
        ht_pv++;
        if (ht_pv == Ht) {
            ht_pv = 0;
            head++;
        }
#endif
    }
}
