#!/usr/bin/env python3
"""Factory half of the writer split: descriptor path of the 2D mcast matmul only."""
from pathlib import Path

F = Path("/home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/"
         "matmul_multicore_reuse_mcast_2d_program_factory.cpp")
s = F.read_text()


def sub(old, new, count=1):
    global s
    if new in s and old not in s:
        print("  already patched")
        return
    n = s.count(old)
    assert n >= count, "anchor missing:\n%s" % old[:200]
    s = s.replace(old, new, count)
    print("  ok")


# 1. the opt-in flag, first occurrence = the descriptor path
sub(
    """    const bool output_is_sharded = out_buffer->buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);""",
    """    const bool output_is_sharded = out_buffer->buffer_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    // In the shipped kernels the in1 operand fetch and the output write are the same RISC (BRISC)
    // while the in0 RISC is idle beside it, so the op floors at their serial sum.  Setting
    // TTNN_MM2D_WRITER_ON_IN0=1 moves the write to the in0 RISC.  Unset, nothing below fires: no
    // define, no extra arg, the shipped program.
    const char* writer_on_in0_env_str = std::getenv("TTNN_MM2D_WRITER_ON_IN0");
    const bool writer_on_in0 = writer_on_in0_env_str != nullptr && writer_on_in0_env_str[0] == '1' &&
                               !in0_block_sharded && !output_is_sharded && !fuse_op && !untilize_out;

    bool do_not_inplace_interm0_out_CB = output_is_sharded && (per_core_M != out_block_h);""",
)

# 2. the writer's compile time block, appended to whichever in0 kernel the core runs
sub(
    """    std::vector<uint32_t> in0_sender_compile_time_args;

    uint32_t num_dram_banks = 0;""",
    """    // The writer's compile time block: the ten scalars the in1 writer gets, then the output
    // accessor.  Same values, same order, so the relocated writer is the shipped writer.
    auto append_writer_ct_args = [&](std::vector<uint32_t>& args) {
        args.push_back((std::uint32_t)1);                   // out_tensor_stride_w
        args.push_back((std::uint32_t)N);                   // out_tensor_stride_h
        args.push_back((std::uint32_t)out_subblock_w);      // out_tensor_next_subblock_stride_w
        args.push_back((std::uint32_t)out_subblock_h * N);  // out_tensor_next_subblock_stride_h
        args.push_back((std::uint32_t)out_block_w);         // out_tensor_next_w_dim_block_stride
        args.push_back((std::uint32_t)out_block_h * N);     // out_tensor_next_h_dim_block_stride
        args.push_back((std::uint32_t)out_subblock_w);
        args.push_back((std::uint32_t)out_subblock_h);
        args.push_back((std::uint32_t)(out_subblock_w * out_subblock_h));
        args.push_back((std::uint32_t)M * N);  // MtNt
        tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(args);
    };

    std::vector<uint32_t> in0_sender_compile_time_args;

    uint32_t num_dram_banks = 0;""",
)

sub(
    """    in0_sender_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
""",
    """    in0_sender_compile_time_args.push_back((std::uint32_t)(fuse_op && fused_op_signaler->is_all_gather()));
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
    if (writer_on_in0) {
        append_writer_ct_args(in0_sender_compile_time_args);
    }
""",
)

sub(
    """        (std::uint32_t)B,     // batch
        (std::uint32_t)false  // get_batch_from_reader
    };
    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {""",
    """        (std::uint32_t)B,     // batch
        (std::uint32_t)false  // get_batch_from_reader
    };
    if (writer_on_in0) {
        append_writer_ct_args(in0_receiver_compile_time_args);
    }
    std::vector<uint32_t> in1_receiver_writer_compile_time_args = {""",
)

# 3. defines
sub(
    """    // Helper to convert std::map defines to KernelDescriptor::Defines (vector of pairs)""",
    """    if (writer_on_in0) {
        mm_kernel_in0_sender_interleaved_defines["WRITER_ON_IN0"] = "1";
        mm_kernel_in1_sender_writer_defines["WRITER_OFF_IN1"] = "1";
        mm_kernel_in1_receiver_writer_defines["WRITER_OFF_IN1"] = "1";
        mm_kernel_in1_receiver_writer_other_noc_setup_defines["WRITER_OFF_IN1"] = "1";
    }

    // Helper to convert std::map defines to KernelDescriptor::Defines (vector of pairs)""",
)

# 4. the in0 kernels need cb_out, and the two receiver descriptors need the define too
sub(
    """        in0_sender_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in0_sharded", tt::CBIndex::c_2},
            {"cb_sparsity", tt::CBIndex::c_6},
            {"cb_in0_intermediate", tt::CBIndex::c_8},
        };""",
    """        in0_sender_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
            {"cb_in0_sharded", tt::CBIndex::c_2},
            {"cb_sparsity", tt::CBIndex::c_6},
            {"cb_in0_intermediate", tt::CBIndex::c_8},
        };
        if (writer_on_in0) {
            in0_sender_kernel_desc.named_compile_time_args.push_back({"cb_out", tt::CBIndex::c_4});
        }""",
)

sub(
    """        in0_receiver_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
        };""",
    """        in0_receiver_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
        };
        if (writer_on_in0) {
            in0_receiver_kernel_desc.named_compile_time_args.push_back({"cb_out", tt::CBIndex::c_4});
            in0_receiver_kernel_desc.defines.push_back({"WRITER_ON_IN0", "1"});
        }""",
)

sub(
    """        in0_receiver_other_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_other_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
        };""",
    """        in0_receiver_other_kernel_desc.compile_time_args = in0_receiver_compile_time_args;
        in0_receiver_other_kernel_desc.named_compile_time_args = {
            {"cb_in0", tt::CBIndex::c_0},
        };
        if (writer_on_in0) {
            in0_receiver_other_kernel_desc.named_compile_time_args.push_back({"cb_out", tt::CBIndex::c_4});
            in0_receiver_other_kernel_desc.defines.push_back({"WRITER_ON_IN0", "1"});
        }""",
)

# 5. per-core writer runtime args
sub(
    """    uint32_t in0_end_idx = num_blocks_y - 1;
    uint32_t in1_end_idx = num_blocks_x - 1;

    for (const auto& core : cores) {
        CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};""",
    """    uint32_t in0_end_idx = num_blocks_y - 1;
    uint32_t in1_end_idx = num_blocks_x - 1;

    // The writer's per-core runtime args: the values the in1 receiver writer gets, with a leading
    // enable flag so an in0 core without matmul work never touches the output CB.
    auto writer_rt_args = [&](uint32_t in0_idx, uint32_t in1_idx, bool has_work) {
        const bool last_h = in0_idx == in0_end_idx;
        const bool last_w = in1_idx == in1_end_idx;
        std::vector<uint32_t> a;
        a.push_back((std::uint32_t)(has_work ? 1 : 0));
        a.push_back((std::uint32_t)out_buffer->address());
        a.push_back(((std::uint32_t)in1_idx * per_core_N) + (in0_idx * per_core_M * N));
        a.push_back((std::uint32_t)(out_block_h / out_subblock_h));
        a.push_back((std::uint32_t)(last_h ? last_block_num_nonzero_subblocks_h : out_block_h / out_subblock_h));
        a.push_back((std::uint32_t)(last_h ? last_subblock_of_last_block_h : out_subblock_h));
        a.push_back((std::uint32_t)(last_h ? last_block_padded_block_tiles_h_skip : 0));
        a.push_back((std::uint32_t)(out_block_w / out_subblock_w));
        a.push_back((std::uint32_t)(last_w ? last_block_num_nonzero_subblocks_w : out_block_w / out_subblock_w));
        a.push_back((std::uint32_t)(last_w ? last_subblock_of_last_block_w : out_subblock_w));
        a.push_back((std::uint32_t)(last_w ? last_block_padded_subblock_tiles_addr_skip : 0));
        a.push_back((std::uint32_t)(last_w ? last_block_padded_block_tiles_w_skip : 0));
        a.push_back((std::uint32_t)(last_h ? last_out_num_blocks_h : out_num_blocks_y));
        a.push_back((std::uint32_t)(last_w ? last_out_num_blocks_w : out_num_blocks_x));
        return a;
    };

    for (const auto& core : cores) {
        CoreCoord left_core = {(std::size_t)start_core_x, (std::size_t)core.y};""",
)

sub(
    """            {
                std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> in0_args(
                    mm_in0_sender_args.begin(), mm_in0_sender_args.end());
                in0_args[0] = in0_buffer;
                in0_sender_kernel_desc.emplace_runtime_args(core, in0_args);
            }""",
    """            {
                std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> in0_args(
                    mm_in0_sender_args.begin(), mm_in0_sender_args.end());
                in0_args[0] = in0_buffer;
                if (writer_on_in0) {
                    const size_t base = in0_args.size();
                    const auto w = writer_rt_args(in0_idx, in1_idx, in0_idx < num_blocks_y and in1_idx < num_blocks_x);
                    in0_args.insert(in0_args.end(), w.begin(), w.end());
                    in0_args[base + 1] = out_buffer;
                }
                in0_sender_kernel_desc.emplace_runtime_args(core, in0_args);
            }""",
)

sub(
    """            // left half
            if (core.x <= half_core || (!transpose_mcast and core.y == start_core_y)) {
                in0_receiver_kernel_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
            }
            // right half
            else {
                in0_receiver_other_kernel_desc.runtime_args.emplace_back(core, mm_in0_receiver_args);
            }""",
    """            std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> in0_recv_args(
                mm_in0_receiver_args.begin(), mm_in0_receiver_args.end());
            if (writer_on_in0) {
                const size_t base = in0_recv_args.size();
                const auto w = writer_rt_args(in0_idx, in1_idx, in0_idx < num_blocks_y and in1_idx < num_blocks_x);
                in0_recv_args.insert(in0_recv_args.end(), w.begin(), w.end());
                in0_recv_args[base + 1] = out_buffer;
            }
            // left half
            if (core.x <= half_core || (!transpose_mcast and core.y == start_core_y)) {
                in0_receiver_kernel_desc.emplace_runtime_args(core, in0_recv_args);
            }
            // right half
            else {
                in0_receiver_other_kernel_desc.emplace_runtime_args(core, in0_recv_args);
            }""",
)

F.write_text(s)
print("factory done")
