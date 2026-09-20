#!/usr/bin/env python3
"""Writer split for tt-metal v0.67.4, the version tt-bio's ttnn wheel is built from.

`trix-writer-split` built this against tt-metal 2026-05-16 (v0.71.0-dev), whose matmul factory
uses the ProgramDescriptor API and whose kernels use the `Noc`/`CircularBuffer` wrappers.  Neither
exists in v0.67.4, so both halves are rewritten here against the classic
`CreateKernel`/`SetRuntimeArgs` API.  The mechanism is identical: under TTNN_MM2D_WRITER_ON_IN0=1
the output write is compiled out of the in1 sender/receiver kernels (BRISC) and into the in0
sender kernel (NCRISC), so the in1 fetch and the output drain are two threads instead of one.

Live path on this version is `process_mcast_in1_program_and_create_override_variables` in the 1D
factory, the classic-API counterpart of the descriptor builder the previous row proved with a
stderr marker.  A marker is kept here under TTNN_MM_PATH_MARKER=1 so the firing path is proven on
this build too rather than inherited.

With the env unset: no define, no extra compile time arg, no extra runtime arg, the shipped program.
"""
import sys
from pathlib import Path

TREE = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ttuser/tt-metal-0674")
ROOT = TREE / "ttnn/cpp/ttnn/operations/matmul/device"
KD = ROOT / "kernels/dataflow"
F1D = ROOT / "factory/matmul_multicore_reuse_mcast_1d_program_factory.cpp"
HPP = Path(__file__).resolve().parent / "matmul_out_writer_in0_0674.hpp"


def edit(path, old, new, count=1):
    s = path.read_text()
    if new in s and old not in s:
        print("  already patched: %s" % path.name)
        return
    n = s.count(old)
    assert n == count, "anchor found %d times (want %d) in %s:\n%s" % (n, count, path.name, old[:220])
    path.write_text(s.replace(old, new, count))
    print("  patched %s" % path.name)


# ---------------------------------------------------------------- the writer, as a header
target = KD / "matmul_out_writer_in0.hpp"
target.write_text(HPP.read_text())
print("  installed %s" % target)

# ---------------------------------------------------------------- in1 kernels: writer compiled out
edit(
    KD / "reader_bmm_tile_layout_in1_sender_writer_padding.cpp",
    """#ifndef OUT_SHARDED
                    // WRITER
                    uint32_t num_blocks_w_dim_ =""",
    """#if !defined(OUT_SHARDED) && !defined(WRITER_OFF_IN1)
                    // WRITER
                    uint32_t num_blocks_w_dim_ =""",
)
edit(
    KD / "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
    """#ifndef OUT_SHARDED
                // WRITER
                uint32_t num_blocks_h_dim_ =""",
    """#if !defined(OUT_SHARDED) && !defined(WRITER_OFF_IN1)
                // WRITER
                uint32_t num_blocks_h_dim_ =""",
)

# ---------------------------------------------------------------- in0 sender: writer compiled in
p = KD / "reader_bmm_tile_layout_in0_sender_padding.cpp"
edit(
    p,
    """#include "ckernel.h"
#include "ckernel_defs.h"
""",
    """#include "ckernel.h"
#include "ckernel_defs.h"
#ifdef WRITER_ON_IN0
#include "matmul_out_writer_in0.hpp"
#endif
""",
)
edit(
    p,
    """    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);
""",
    """    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);
#ifdef WRITER_ON_IN0
    // WRITER, relocated off the in1 RISC
    MatmulOutWriterArgs wargs = matmul_out_writer_args(rt_args_idx);
#endif
""",
)
edit(
    p,
    """    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
""",
    """    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
#ifdef WRITER_ON_IN0
    constexpr uint32_t writer_cta_base = sparsity_args.next_compile_time_args_offset();
    constexpr auto out_args = TensorAccessorArgs<writer_cta_base + 10>();
    constexpr uint32_t cb_id_out = get_named_compile_time_arg_val("cb_out");
#endif
""",
)
edit(
    p,
    """    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);
""",
    """    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);
#ifdef WRITER_ON_IN0
    const auto s_out = TensorAccessor(out_args, wargs.out_tensor_addr, get_tile_size(cb_id_out));
#endif
""",
)
edit(
    p,
    """            uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {""",
    """            uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
#ifdef WRITER_ON_IN0
            uint32_t out_tensor_current_h_dim_block_tile_id = wargs.out_tensor_start_tile_id;
#endif
            for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
#ifdef WRITER_ON_IN0
                uint32_t out_tensor_current_w_dim_block_tile_id = out_tensor_current_h_dim_block_tile_id;
#endif
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {""",
)
edit(
    p,
    """                    }
                }
#ifdef IN0_SHARDED
                in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride_bytes;
#endif  // IN0_SHARDED
                in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
            }""",
    """                    }
#ifdef WRITER_ON_IN0
                    if (wargs.enabled) {
                        matmul_write_out_block<writer_cta_base, cb_id_out, num_blocks_h_dim, num_blocks_w_dim>(
                            s_out, wargs, bh, bw, out_tensor_current_w_dim_block_tile_id);
                    }
                    out_tensor_current_w_dim_block_tile_id += get_compile_time_arg_val(writer_cta_base + 4);
#endif
                }
#ifdef WRITER_ON_IN0
                out_tensor_current_h_dim_block_tile_id += get_compile_time_arg_val(writer_cta_base + 5);
#endif
#ifdef IN0_SHARDED
                in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride_bytes;
#endif  // IN0_SHARDED
                in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
            }""",
)
edit(
    p,
    """            if constexpr (!bcast_A) {
                in0_tensor_start_tile_id += MtKt;
            }
        }
""",
    """            if constexpr (!bcast_A) {
                in0_tensor_start_tile_id += MtKt;
            }
#ifdef WRITER_ON_IN0
            wargs.out_tensor_start_tile_id += get_compile_time_arg_val(writer_cta_base + 9);
#endif
        }
""",
)

# ---------------------------------------------------------------- 1D factory, mcast_in1 path
edit(
    F1D,
    """    std::vector<uint32_t> in0_sender_compile_time_args = {
        // in0 tensor args
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,""",
    """    // The output write shares BRISC with the in1 sender/receiver in the shipped kernels while the
    // in0 RISC sits idle beside it, so the op floors at their serial sum.  TTNN_MM2D_WRITER_ON_IN0=1
    // moves the write to the in0 RISC.  Unset, nothing below fires: no define, no extra arg, the
    // shipped program.
    const char* writer_on_in0_env_str = std::getenv("TTNN_MM2D_WRITER_ON_IN0");
    const bool writer_on_in0 = writer_on_in0_env_str != nullptr && writer_on_in0_env_str[0] == '1' &&
                               !in0_is_sharded && !output_is_sharded && !fuse_op && !untilize_out &&
                               bias_buffer == nullptr;
    if (std::getenv("TTNN_MM_PATH_MARKER") != nullptr) {
        fprintf(
            stderr,
            "MM1D-FLAG mcast_in1 writer_on_in0=%d M=%u N=%u K=%u B=%u\\n",
            (int)writer_on_in0,
            (unsigned)M,
            (unsigned)N,
            (unsigned)K,
            (unsigned)B);
    }

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

    std::vector<uint32_t> in0_sender_compile_time_args = {
        // in0 tensor args
        (std::uint32_t)in0_tensor_stride_w,
        (std::uint32_t)in0_tensor_stride_h,
        (std::uint32_t)in0_tensor_next_block_stride,""",
)
edit(
    F1D,
    """    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {""",
    """    in0_sender_compile_time_args.push_back((std::uint32_t)fuse_op);
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(in0_sender_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(in0_sender_compile_time_args);  // placeholder for sparsity
    if (writer_on_in0) {
        append_writer_ct_args(in0_sender_compile_time_args);
    }

    std::vector<uint32_t> in1_sender_writer_compile_time_args = {""",
)
edit(
    F1D,
    """    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_defines,
            .named_compile_args = {
                {"cb_in0", tt::CBIndex::c_0},
                {"cb_in0_sharded", tt::CBIndex::c_2},
                {"cb_sparsity", tt::CBIndex::c_6},
                {"cb_in0_intermediate", tt::CBIndex::c_8},
            }});""",
    """    if (writer_on_in0) {
        mm_kernel_in0_sender_defines["WRITER_ON_IN0"] = "1";
        mm_kernel_in1_sender_writer_defines["WRITER_OFF_IN1"] = "1";
        mm_kernel_in1_receiver_writer_defines["WRITER_OFF_IN1"] = "1";
    }

    std::unordered_map<std::string, uint32_t> in0_sender_named_compile_args = {
        {"cb_in0", tt::CBIndex::c_0},
        {"cb_in0_sharded", tt::CBIndex::c_2},
        {"cb_sparsity", tt::CBIndex::c_6},
        {"cb_in0_intermediate", tt::CBIndex::c_8},
    };
    if (writer_on_in0) {
        in0_sender_named_compile_args["cb_out"] = tt::CBIndex::c_4;
    }

    auto mm_kernel_in0_sender_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in0_sender_padding.cpp",
        all_cores,
        tt_metal::DataMovementConfig{
            .processor = tt_metal::DataMovementProcessor::RISCV_1,
            .noc = in0_noc,
            .compile_args = in0_sender_compile_time_args,
            .defines = mm_kernel_in0_sender_defines,
            .named_compile_args = in0_sender_named_compile_args});""",
)
edit(
    F1D,
    """    const auto& cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;""",
    """    // The writer's per-core args in the in1 receiver kernel's layout, which is the general one.
    // The mcast sender core keeps the values the shipped sender writer gets, so the relocated
    // writer does byte-for-byte the same work on every core.
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
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        uint32_t output_idx_x = i / num_blocks_y;
        uint32_t output_idx_y = i % num_blocks_y;""",
)
edit(
    F1D,
    """        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);  // RISCV_1_default""",
    """        if (writer_on_in0) {
            const auto w = writer_rt_args(output_idx_x, output_idx_y, core == start_core);
            mm_in0_sender_args.insert(mm_in0_sender_args.end(), w.begin(), w.end());
        }
        tt_metal::SetRuntimeArgs(program, mm_kernel_in0_sender_id, core, mm_in0_sender_args);  // RISCV_1_default""",
)

# ---------------------------------------------------------------- program cache: out addr override
# The in0 sender now holds the output address in its runtime args, so a cache hit against a
# different output buffer has to update it there too, not only in the in1 kernels.
edit(
    F1D,
    """        auto& reader_runtime_args =
            reader_runtime_args_by_core[override_variables.start_core.x][override_variables.start_core.y];
        reader_runtime_args[0] = src_buffer_a->address();
""",
    """        auto& reader_runtime_args =
            reader_runtime_args_by_core[override_variables.start_core.x][override_variables.start_core.y];
        reader_runtime_args[0] = src_buffer_a->address();
        if (reader_runtime_args.size() >= WRITER_ON_IN0_RT_ARG_COUNT) {
            reader_runtime_args[WRITER_ON_IN0_OUT_ADDR_IDX] = dst_buffer->address();
        }
""",
)
edit(
    F1D,
    """        // in0 sender
        reader_runtime_args[0] = src_buffer_a->address();
        // in1 receiver
        writer_runtime_args[2] = dst_buffer->address();""",
    """        // in0 sender
        reader_runtime_args[0] = src_buffer_a->address();
        if (reader_runtime_args.size() >= WRITER_ON_IN0_RT_ARG_COUNT) {
            reader_runtime_args[WRITER_ON_IN0_OUT_ADDR_IDX] = dst_buffer->address();
        }
        // in1 receiver
        writer_runtime_args[2] = dst_buffer->address();""",
)
edit(
    F1D,
    """inline void override_mcast_in1_program_parameters(""",
    """// The in0 sender takes 8 runtime args shipped; with the writer relocated onto it, 14 more, and
// the output address is the second of those.  fuse_op is excluded from the relocation, so nothing
// else ever grows these args and the size test cannot misfire.
constexpr size_t WRITER_ON_IN0_RT_ARG_COUNT = 8 + 14;
constexpr size_t WRITER_ON_IN0_OUT_ADDR_IDX = 8 + 1;

inline void override_mcast_in1_program_parameters(""",
)

print("0.67.4 writer split applied")
