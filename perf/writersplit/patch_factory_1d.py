#!/usr/bin/env python3
"""Writer split, factory half, for the path key A actually runs.

`create_program_mcast_in1_descriptor` in the 1D factory is the live builder for the trimul key-A
shape (proved with an unfilterable stderr marker: MM1D-PATH descriptor_mcast_in1 fires, both 2D
builders stay silent).  Same five edits as the 2D patch, restricted to that one function, and
simpler because the 1D path has a single in0 kernel covering every core.
"""
from pathlib import Path

F = Path("/home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/factory/"
         "matmul_multicore_reuse_mcast_1d_program_factory.cpp")
s = F.read_text()

FN = "static ProgramDescriptor create_program_mcast_in1_descriptor("
END = "MatmulMultiCoreReuseMcast1DProgramFactory::shared_variables_t matmul_multi_core_reuse_mcast_1d_optimized_("
i0 = s.index(FN)
i1 = s.index(END, i0)
body = s[i0:i1]


def sub(old, new):
    global body
    if new in body and old not in body:
        print("  already patched")
        return
    n = body.count(old)
    assert n == 1, "anchor found %d times:\n%s" % (n, old[:160])
    body = body.replace(old, new, 1)
    print("  ok")


# 1. the opt-in flag and the writer's compile time block
sub(
    """    std::vector<uint32_t> in0_sender_compile_time_args = {""",
    """    // The output write shares BRISC with the in1 sender in the shipped kernels while the in0
    // RISC is idle beside it, so the op floors at their serial sum.  TTNN_MM2D_WRITER_ON_IN0=1
    // moves the write to the in0 RISC.  Unset, nothing below fires: no define, no extra arg, the
    // shipped program.  reuse_in0_in_CB is excluded because then the in0 reader's batch loop and
    // the writer's batch count no longer agree.
    const char* writer_on_in0_env_str = std::getenv("TTNN_MM2D_WRITER_ON_IN0");
    const bool writer_on_in0 = writer_on_in0_env_str != nullptr && writer_on_in0_env_str[0] == '1' &&
                               !in0_is_sharded && !output_is_sharded && !fuse_op && !untilize_out &&
                               !reuse_in0_in_CB;

    // Ten scalars then the output accessor, the same values the in1 writer gets.
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

    std::vector<uint32_t> in0_sender_compile_time_args = {""",
)

sub(
    """    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
""",
    """    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
    if (writer_on_in0) {
        append_writer_ct_args(in0_sender_compile_time_args);
    }
""",
)

# 2. defines, before any kernel descriptor is filled in
sub(
    """    in0_sender_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp";""",
    """    if (writer_on_in0) {
        mm_kernel_in0_sender_defines["WRITER_ON_IN0"] = "1";
        mm_kernel_in1_sender_writer_defines["WRITER_OFF_IN1"] = "1";
        mm_kernel_in1_receiver_writer_defines["WRITER_OFF_IN1"] = "1";
    }

    in0_sender_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp";""",
)

# 3. the in0 kernel needs cb_out
sub(
    """    in0_sender_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in0_sharded", tt::CBIndex::c_2},
        {"cb_sparsity", tt::CBIndex::c_6},
        {"cb_in0_intermediate", tt::CBIndex::c_8},
    };""",
    """    in0_sender_kernel_desc.named_compile_time_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in0_sharded", tt::CBIndex::c_2},
        {"cb_sparsity", tt::CBIndex::c_6},
        {"cb_in0_intermediate", tt::CBIndex::c_8},
    };
    if (writer_on_in0) {
        in0_sender_kernel_desc.named_compile_time_args.push_back({"cb_out", tt::CBIndex::c_4});
    }""",
)

# 4. per-core writer runtime args
sub(
    """    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {""",
    """    // The writer's per-core args in the in1 receiver kernel's layout, which is the general one.
    // The mcast sender core keeps the values the shipped sender writer gets, so the relocated
    // writer is byte-for-byte the same work on every core.
    auto writer_rt_args = [&](uint32_t output_idx_x, uint32_t output_idx_y, bool sender_core) {
        const bool last_h = !sender_core && output_idx_y == num_blocks_y - 1;
        std::vector<uint32_t> a;
        a.push_back((std::uint32_t)1);  // enabled
        a.push_back((std::uint32_t)out_buffer->address());
        a.push_back(((std::uint32_t)output_idx_x * per_core_N) + (output_idx_y * per_core_M * N));
        a.push_back((std::uint32_t)(out_block_h / out_subblock_h));
        a.push_back((std::uint32_t)(last_h ? last_block_num_nonzero_subblocks_h : out_block_h / out_subblock_h));
        a.push_back((std::uint32_t)(last_h ? last_subblock_of_last_block_h : out_subblock_h));
        a.push_back((std::uint32_t)(last_h ? last_block_padded_block_tiles_h_skip : 0));
        a.push_back((std::uint32_t)(out_block_w / out_subblock_w));
        a.push_back((std::uint32_t)(out_block_w / out_subblock_w));
        a.push_back((std::uint32_t)out_subblock_w);
        a.push_back((std::uint32_t)0);
        a.push_back((std::uint32_t)0);
        a.push_back((std::uint32_t)(last_h ? last_out_num_blocks_h : out_num_blocks_y));
        a.push_back((std::uint32_t)out_num_blocks_x);
        return a;
    };

    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {""",
)

sub(
    """            std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> in0_sender_variant(
                mm_in0_sender_args.begin(), mm_in0_sender_args.end());
            in0_sender_variant[0] = in0_buffer;
            in0_sender_kernel_desc.emplace_runtime_args(core, in0_sender_variant);""",
    """            std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> in0_sender_variant(
                mm_in0_sender_args.begin(), mm_in0_sender_args.end());
            in0_sender_variant[0] = in0_buffer;
            if (writer_on_in0) {
                const size_t base = in0_sender_variant.size();
                const auto w = writer_rt_args(output_idx_x, output_idx_y, core == start_core);
                in0_sender_variant.insert(in0_sender_variant.end(), w.begin(), w.end());
                in0_sender_variant[base + 1] = out_buffer;
            }
            in0_sender_kernel_desc.emplace_runtime_args(core, in0_sender_variant);""",
)

F.write_text(s[:i0] + body + s[i1:])
print("1D factory done")
