"""Software-pipelined writer: wait for each out-subblock as soon as the packer pushes it (so the
writes start early, as the stock kernel does), but hold the write barrier DEPTH subblocks behind,
so DEPTH * 8 KB is in flight to DRAM instead of 8 KB.

Stock kernel == DEPTH 1. Guarded to the unpadded case; anything else runs the original code."""
import re, sys

DEPTH = int(sys.argv[1])

FAST = """{ind}const uint32_t dws_nsb = out_num_nonzero_subblocks_h_ * out_num_nonzero_subblocks_w_;
{ind}const bool dws_flat =
{ind}    (out_num_nonzero_subblocks_h_ == out_num_nonzero_subblocks_h) &&
{ind}    (out_num_nonzero_subblocks_w_ == out_num_nonzero_subblocks_w) &&
{ind}    (out_last_subblock_h == out_subblock_h) && (out_last_subblock_w == out_subblock_w) &&
{ind}    (padded_subblock_tiles_addr_skip == 0) && (padded_block_tiles_w_skip == 0) &&
{ind}    (padded_block_tiles_h_skip == 0) && ({wrguard});
{ind}if (dws_flat) {{
{ind}    constexpr uint32_t dws_depth = {D}u;
{ind}    const uint32_t dws_sb_bytes = out_subblock_tile_count * output_single_tile_size_bytes;
{ind}    uint32_t dws_k = 0;
{ind}    uint32_t dws_off = 0;
{ind}    for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {{
{ind}        uint32_t sbw_id = out_tensor_sbh_start_tile_id;
{ind}        for (uint32_t sbw = 0; sbw < out_num_nonzero_subblocks_w_; ++sbw) {{
{ind}            cb_out.wait_front(out_subblock_tile_count * (dws_k + 1));
{ind}            uint32_t row_id = sbw_id;
{ind}            for (uint32_t h = 0; h < out_subblock_h; ++h) {{
{ind}                uint32_t tid = row_id;
{ind}                for (uint32_t w = 0; w < out_subblock_w; ++w) {{
{ind}                    noc.async_write(
{ind}                        use<CircularBuffer::AddrSelector::READ_PTR>(cb_out),
{ind}                        s,
{ind}                        output_single_tile_size_bytes,
{ind}                        {{.offset_bytes = dws_off}},
{ind}                        {{.page_id = tid}});
{ind}                    dws_off += output_single_tile_size_bytes;
{ind}                    tid += out_tensor_stride_w;
{ind}                }}
{ind}                row_id += out_tensor_stride_h;
{ind}            }}
{ind}            ++dws_k;
{ind}            if (dws_k == dws_depth) {{
{ind}                noc.async_write_barrier();
{ind}                cb_out.pop_front(out_subblock_tile_count * dws_k);
{ind}                dws_k = 0;
{ind}                dws_off = 0;
{ind}            }}
{ind}            sbw_id += out_tensor_next_subblock_stride_w;
{ind}        }}
{ind}        out_tensor_sbh_start_tile_id += out_tensor_next_subblock_stride_h;
{ind}    }}
{ind}    if (dws_k != 0) {{
{ind}        noc.async_write_barrier();
{ind}        cb_out.pop_front(out_subblock_tile_count * dws_k);
{ind}    }}
{ind}}} else {{
"""

def patch(path, wrguard):
    lines = open(path + ".dws_backup").read().split("\n")
    start = next(i for i, l in enumerate(lines)
                 if "for (uint32_t sbh = 0; sbh < out_num_nonzero_subblocks_h_; ++sbh) {" in l)
    end = next(i for i, l in enumerate(lines) if i > start and "Pop row(s) of fully padded subblocks" in l)
    close = next(i for i, l in enumerate(lines) if i > end and l.strip() == "}")
    ind = re.match(r"\s*", lines[start]).group(0)
    fast = FAST.format(ind=ind, wrguard=wrguard, D=DEPTH).rstrip("\n").split("\n")
    open(path, "w").write("\n".join(lines[:start] + fast + lines[start:close + 1] + [ind + "}"] + lines[close + 1:]))

D = "/home/ttuser/tt-metal/ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/"
patch(D + "reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
      "bh < num_blocks_h_dim_ && bw < num_blocks_w_dim_")
patch(D + "reader_bmm_tile_layout_in1_sender_writer_padding.cpp", "bw < num_blocks_w_dim_")
print("pipelined DEPTH=%d" % DEPTH)
