// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Output writer for the 2D mcast matmul, moved off the in1 sender RISC (BRISC) onto the in0 RISC
// (NCRISC).  In the shipped kernels the in1 operand fetch and the output drain are the same
// thread, so the op floors at their serial sum while the in0 RISC is idle beside it.  With
// WRITER_ON_IN0 the in1 kernels keep the operand fetch and mcast and compile their writer out,
// and the in0 kernels call this instead.  The body is the shipped writer from
// reader_bmm_tile_layout_in1_receiver_writer_padding.cpp, unchanged apart from being a function.
//
// CTA is the first compile time arg index of the writer block: ten scalars followed by the
// output TensorAccessorArgs.

#pragma once

struct MatmulOutWriterArgs {
    uint32_t enabled;
    uint32_t out_tensor_addr;
    uint32_t out_tensor_start_tile_id;
    uint32_t out_num_nonzero_subblocks_h;
    uint32_t out_last_num_nonzero_subblocks_h;
    uint32_t out_last_subblock_h;
    uint32_t padded_block_tiles_h_skip;
    uint32_t out_num_nonzero_subblocks_w;
    uint32_t out_last_num_nonzero_subblocks_w;
    uint32_t out_last_subblock_w;
    uint32_t padded_subblock_tiles_addr_skip;
    uint32_t padded_block_tiles_w_skip;
    uint32_t last_num_blocks_h_dim;
    uint32_t last_num_blocks_w_dim;
};

FORCE_INLINE MatmulOutWriterArgs matmul_out_writer_args(uint32_t& rt_args_idx) {
    MatmulOutWriterArgs a;
    a.enabled = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_tensor_addr = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_tensor_start_tile_id = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_last_num_nonzero_subblocks_h = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_last_subblock_h = get_arg_val<uint32_t>(rt_args_idx++);
    a.padded_block_tiles_h_skip = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_last_num_nonzero_subblocks_w = get_arg_val<uint32_t>(rt_args_idx++);
    a.out_last_subblock_w = get_arg_val<uint32_t>(rt_args_idx++);
    a.padded_subblock_tiles_addr_skip = get_arg_val<uint32_t>(rt_args_idx++);
    a.padded_block_tiles_w_skip = get_arg_val<uint32_t>(rt_args_idx++);
    a.last_num_blocks_h_dim = get_arg_val<uint32_t>(rt_args_idx++);
    a.last_num_blocks_w_dim = get_arg_val<uint32_t>(rt_args_idx++);
    return a;
}

template <uint32_t CTA, uint32_t num_blocks_h_dim, uint32_t num_blocks_w_dim, typename Accessor>
FORCE_INLINE void matmul_write_out_block(
    Noc& noc,
    CircularBuffer& cb_out,
    const Accessor& s,
    const MatmulOutWriterArgs& a,
    uint32_t bh,
    uint32_t bw,
    uint32_t out_tensor_current_w_dim_block_tile_id) {
    constexpr uint32_t out_tensor_stride_w = get_compile_time_arg_val(CTA + 0);
    constexpr uint32_t out_tensor_stride_h = get_compile_time_arg_val(CTA + 1);
    constexpr uint32_t out_tensor_next_subblock_stride_w = get_compile_time_arg_val(CTA + 2);
    constexpr uint32_t out_tensor_next_subblock_stride_h = get_compile_time_arg_val(CTA + 3);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(CTA + 6);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(CTA + 7);
    constexpr uint32_t out_subblock_tile_count = get_compile_time_arg_val(CTA + 8);
    constexpr uint32_t cb_id_out0 = get_named_compile_time_arg_val("cb_out");
    const uint32_t output_single_tile_size_bytes = get_tile_size(cb_id_out0);

    uint32_t num_blocks_h_dim_ = bh >= a.last_num_blocks_h_dim - 1 ? a.last_num_blocks_h_dim : num_blocks_h_dim;
    uint32_t num_blocks_w_dim_ = bw >= a.last_num_blocks_w_dim - 1 ? a.last_num_blocks_w_dim : num_blocks_w_dim;
    uint32_t out_num_nonzero_subblocks_h_ = a.out_num_nonzero_subblocks_h;
    uint32_t out_num_nonzero_subblocks_w_ = a.out_num_nonzero_subblocks_w;
    if (bh == num_blocks_h_dim_ - 1) {
        out_num_nonzero_subblocks_h_ = a.out_last_num_nonzero_subblocks_h;
    }
    if (bw == num_blocks_w_dim_ - 1) {
        out_num_nonzero_subblocks_w_ = a.out_last_num_nonzero_subblocks_w;
    }
    uint32_t out_tensor_sbh_start_tile_id = out_tensor_current_w_dim_block_tile_id;
    for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {
        uint32_t out_tensor_sbw_start_tile_id = out_tensor_sbh_start_tile_id;
        for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {
            uint32_t out_tensor_sb_row_start_tile_id = out_tensor_sbw_start_tile_id;

            uint32_t out_subblock_h_ = out_subblock_h;
            uint32_t out_subblock_w_ = out_subblock_w;
            uint32_t subblock_tiles_addr_skip = 0;
            if (bh == num_blocks_h_dim_ - 1 && sbh == out_num_nonzero_subblocks_h_ - 1) {
                out_subblock_h_ = a.out_last_subblock_h;
            }
            if (bw == num_blocks_w_dim_ - 1 && sbw == out_num_nonzero_subblocks_w_ - 1) {
                out_subblock_w_ = a.out_last_subblock_w;
                subblock_tiles_addr_skip = a.padded_subblock_tiles_addr_skip;
            }

            cb_out.wait_front(out_subblock_tile_count);
            uint32_t out_read_offset = 0;

            for (uint32_t h = 0; h < out_subblock_h_; ++h) {
                uint32_t out_tensor_tile_id = out_tensor_sb_row_start_tile_id;
                for (uint32_t w = 0; w < out_subblock_w_; ++w) {
                    if (bh < num_blocks_h_dim_ && bw < num_blocks_w_dim_) {
                        noc.async_write(
                            use<CircularBuffer::AddrSelector::READ_PTR>(cb_out),
                            s,
                            output_single_tile_size_bytes,
                            {.offset_bytes = out_read_offset},
                            {.page_id = out_tensor_tile_id});
                    }

                    out_read_offset += output_single_tile_size_bytes;

                    out_tensor_tile_id += out_tensor_stride_w;
                }
                // Skip padded tiles in subblock along row
                out_read_offset += subblock_tiles_addr_skip;
                out_tensor_sb_row_start_tile_id += out_tensor_stride_h;
            }

            noc.async_write_barrier();

            cb_out.pop_front(out_subblock_tile_count);
            out_tensor_sbw_start_tile_id += out_tensor_next_subblock_stride_w;
        }
        // Pop fully padded subblocks along the row
        if (bw == num_blocks_w_dim_ - 1) {
            cb_out.wait_front(a.padded_block_tiles_w_skip);
            cb_out.pop_front(a.padded_block_tiles_w_skip);
        }
        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
    }
    // Pop row(s) of fully padded subblocks
    if (bh == num_blocks_h_dim_ - 1) {
        cb_out.wait_front(a.padded_block_tiles_h_skip);
        cb_out.pop_front(a.padded_block_tiles_h_skip);
    }
}
