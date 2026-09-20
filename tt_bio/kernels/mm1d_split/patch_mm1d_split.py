#!/usr/bin/env python3
"""Generate `tt_bio/kernels/mm1d_split/` from the wheel's own 1D mcast matmul dataflow kernels.

The writer split: the shipped kernels run the output write on the same BRISC as the in1
sender/receiver while the in0 RISC sits idle beside it, so the op floors at their serial sum.
Under `WRITER_ON_IN0` the write is compiled out of the in1 kernels and into the in0 sender, and
the two become parallel threads.

`perf/writersplit/patch_0674.py` on wk/trix-writer-split-build (commit b28bfd6dc) is the source
of truth for the mechanism; these are the same kernel-side edits, emitted into our own directory
instead of patched into a tt-metal checkout, because tt-bio ships against the pip wheel and a
source patch cannot reach a user.  The host half lives in `tt_bio/mm1d_generic.py`.

Both arms are guarded, so with no macro defined the generated files compile to the wheel's own
kernels.  Run from the repo root; overwrites the generated files.
"""

import sys
from pathlib import Path

OUT = Path("tt_bio/kernels/mm1d_split")

IN0_SENDER = "reader_bmm_tile_layout_in0_sender_padding.cpp"
IN1_SENDER = "reader_bmm_tile_layout_in1_sender_writer_padding.cpp"
IN1_RECEIVER = "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp"

# The in1 kernels: the write is compiled out.
IN1_SENDER_EDITS = [(
    """#ifndef OUT_SHARDED
                    // WRITER
                    uint32_t num_blocks_w_dim_ =""",
    """#if !defined(OUT_SHARDED) && !defined(WRITER_OFF_IN1)
                    // WRITER
                    uint32_t num_blocks_w_dim_ =""")]

IN1_RECEIVER_EDITS = [(
    """#ifndef OUT_SHARDED
                // WRITER
                uint32_t num_blocks_h_dim_ =""",
    """#if !defined(OUT_SHARDED) && !defined(WRITER_OFF_IN1)
                // WRITER
                uint32_t num_blocks_h_dim_ =""")]

# The in0 sender: the write is compiled in.  `writer_cta_base` is taken from the sparsity
# accessor's own offset rather than a literal, so a wheel that grows an arg ahead of it still
# lands the writer's args in the right place.
IN0_SENDER_EDITS = [
    ("""#include "ckernel.h"
#include "ckernel_defs.h"
""",
     """#include "ckernel.h"
#include "ckernel_defs.h"
#ifdef WRITER_ON_IN0
#include "matmul_out_writer_in0.hpp"
#endif
"""),
    ("""    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);
""",
     """    // sparsity args
    const uint32_t sparsity_addr = get_arg_val<uint32_t>(rt_args_idx++);
#ifdef WRITER_ON_IN0
    // WRITER, relocated off the in1 RISC
    MatmulOutWriterArgs wargs = matmul_out_writer_args(rt_args_idx);
#endif
"""),
    ("""    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
""",
     """    constexpr auto sparsity_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
#ifdef WRITER_ON_IN0
    constexpr uint32_t writer_cta_base = sparsity_args.next_compile_time_args_offset();
    constexpr auto out_args = TensorAccessorArgs<writer_cta_base + 10>();
    constexpr uint32_t cb_id_out = get_named_compile_time_arg_val("cb_out");
#endif
"""),
    ("""    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);
""",
     """    const auto s_sparsity = TensorAccessor(sparsity_args, sparsity_addr, sparsity_pagesize);
#ifdef WRITER_ON_IN0
    const auto s_out = TensorAccessor(out_args, wargs.out_tensor_addr, get_tile_size(cb_id_out));
#endif
"""),
    ("""            uint32_t in0_tensor_current_h_dim_block_tile_id = in0_tensor_start_tile_id;
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
                for (uint32_t bw = 0; bw < num_blocks_w_dim; ++bw) {"""),
    ("""                    }
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
            }"""),
    ("""            if constexpr (!bcast_A) {
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
"""),
]


def apply(text, edits, path):
    for old, new in edits:
        if text.count(old) != 1:
            raise SystemExit("patch site not unique in %s: %r" % (path, old[:80]))
        text = text.replace(old, new)
    return text


def main():
    sys.path.insert(0, ".")
    from tt_bio.mm1d_generic import wheel_dataflow_dir
    src = wheel_dataflow_dir()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, edits in ((IN0_SENDER, IN0_SENDER_EDITS),
                        (IN1_SENDER, IN1_SENDER_EDITS),
                        (IN1_RECEIVER, IN1_RECEIVER_EDITS)):
        (OUT / name).write_text(apply((src / name).read_text(), edits, name))
    print("wrote", OUT, "from", src)


if __name__ == "__main__":
    main()
