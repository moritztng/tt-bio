#!/usr/bin/env python3
"""Move the 2D mcast matmul's output write off the in1 sender RISC onto the in0 RISC.

Applies to /home/ttuser/tt-metal in place.  Every edit is anchored on exact text and asserts the
anchor was found exactly where expected, so a re-run on an already patched tree is a no-op and a
drifted tree is a hard error rather than a silent half-patch.  Behaviour is unchanged unless
TTNN_MM2D_WRITER_ON_IN0=1: with the env unset no define is added, no arg is appended, and the
JIT hash of every kernel is the shipped one.
"""
import sys
from pathlib import Path

ROOT = Path("/home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/matmul/device")
KD = ROOT / "kernels/dataflow"
FACTORY = ROOT / "factory/matmul_multicore_reuse_mcast_2d_program_factory.cpp"


def edit(path, old, new, count=1):
    s = path.read_text()
    if new in s and old not in s:
        print("  already patched: %s" % path.name)
        return
    n = s.count(old)
    assert n == count, "anchor found %d times (want %d) in %s:\n%s" % (n, count, path.name, old[:200])
    path.write_text(s.replace(old, new, count))
    print("  patched %s" % path.name)


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

# ---------------------------------------------------------------- in0 receiver: writer compiled in
p = KD / "reader_bmm_tile_layout_in0_receiver.cpp"
edit(
    p,
    """#include "api/dataflow/noc_semaphore.h"

void kernel_main() {""",
    """#include "api/dataflow/noc_semaphore.h"
#ifdef WRITER_ON_IN0
#include "api/tensor/noc_traits.h"
#include "matmul_out_writer_in0.hpp"
#endif

void kernel_main() {""",
)
edit(
    p,
    """    const uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(1);
""",
    """    const uint32_t in0_mcast_sender_noc_y = get_arg_val<uint32_t>(1);
#ifdef WRITER_ON_IN0
    uint32_t rt_args_idx = 2;
    MatmulOutWriterArgs wargs = matmul_out_writer_args(rt_args_idx);
#endif
""",
)
edit(
    p,
    """    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);""",
    """    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
#ifdef WRITER_ON_IN0
    constexpr uint32_t writer_cta_base = 8;
    constexpr auto out_args = TensorAccessorArgs<writer_cta_base + 10>();
    constexpr uint32_t cb_id_out = get_named_compile_time_arg_val("cb_out");
#endif

    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
#ifdef WRITER_ON_IN0
    CircularBuffer cb_out(cb_id_out);
    const auto s_out = TensorAccessor(out_args, wargs.out_tensor_addr);
#endif""",
)
edit(
    p,
    """        for (uint32_t bh = 0; bh < num_blocks_h_dim; ++bh) {
            for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {""",
    """#ifdef WRITER_ON_IN0
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
    """                    cb_in0.push_back(in0_block_num_tiles);
                }
            }
        }
    }
}""",
    """                    cb_in0.push_back(in0_block_num_tiles);
                }
#ifdef WRITER_ON_IN0
                if (wargs.enabled) {
                    matmul_write_out_block<writer_cta_base, num_blocks_h_dim, num_blocks_w_dim>(
                        noc, cb_out, s_out, wargs, bh, bw, out_tensor_current_w_dim_block_tile_id);
                }
                out_tensor_current_w_dim_block_tile_id += get_compile_time_arg_val(writer_cta_base + 4);
#endif
            }
#ifdef WRITER_ON_IN0
            out_tensor_current_h_dim_block_tile_id += get_compile_time_arg_val(writer_cta_base + 5);
#endif
        }
#ifdef WRITER_ON_IN0
        wargs.out_tensor_start_tile_id += get_compile_time_arg_val(writer_cta_base + 9);
#endif
    }
}""",
)

# ---------------------------------------------------------------- in0 sender: writer compiled in
p = KD / "reader_bmm_tile_layout_in0_sender_padding.cpp"
edit(
    p,
    """#include "api/core_local_mem.h"

void kernel_main() {""",
    """#include "api/core_local_mem.h"
#ifdef WRITER_ON_IN0
#include "matmul_out_writer_in0.hpp"
#endif

void kernel_main() {""",
)
edit(
    p,
    """    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);""",
    """#ifdef WRITER_ON_IN0
    constexpr uint32_t writer_cta_base = sparsity_args.next_compile_time_args_offset();
    constexpr auto out_args = TensorAccessorArgs<writer_cta_base + 10>();
    constexpr uint32_t cb_id_out = get_named_compile_time_arg_val("cb_out");
    MatmulOutWriterArgs wargs = matmul_out_writer_args(rt_args_idx);
#endif

    constexpr uint32_t cb_id_in0 = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in0_single_tile_size_bytes = get_tile_size(cb_id_in0);""",
)
edit(
    p,
    """    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);""",
    """    Noc noc;
    CircularBuffer cb_in0(cb_id_in0);
#ifdef WRITER_ON_IN0
    CircularBuffer cb_out(cb_id_out);
    const auto s_out = TensorAccessor(out_args, wargs.out_tensor_addr);
#endif""",
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
            }

            if constexpr (!bcast_A) {
                in0_tensor_start_tile_id += MtKt;
            }
        }""",
    """                    }
#ifdef WRITER_ON_IN0
                    if (wargs.enabled) {
                        matmul_write_out_block<writer_cta_base, num_blocks_h_dim, num_blocks_w_dim>(
                            noc, cb_out, s_out, wargs, bh, bw, out_tensor_current_w_dim_block_tile_id);
                    }
                    out_tensor_current_w_dim_block_tile_id += get_compile_time_arg_val(writer_cta_base + 4);
#endif
                }
#ifdef IN0_SHARDED
                in0_tensor_current_h_dim_block_start_addr += in0_tensor_next_h_dim_block_stride_bytes;
#endif  // IN0_SHARDED
                in0_tensor_current_h_dim_block_tile_id += in0_tensor_next_h_dim_block_stride;
#ifdef WRITER_ON_IN0
                out_tensor_current_h_dim_block_tile_id += get_compile_time_arg_val(writer_cta_base + 5);
#endif
            }
#ifdef WRITER_ON_IN0
            wargs.out_tensor_start_tile_id += get_compile_time_arg_val(writer_cta_base + 9);
#endif

            if constexpr (!bcast_A) {
                in0_tensor_start_tile_id += MtKt;
            }
        }""",
)

print("kernels done")
sys.exit(0)
