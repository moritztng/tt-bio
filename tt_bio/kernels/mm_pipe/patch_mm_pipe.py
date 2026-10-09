"""mm_pipe: software-pipeline the output writes of ttnn's 2D-mcast matmul behind the next block's in1 transfer.

ttnn's in1 sender/writer and in1 receiver/writer kernels (BRISC) run, per output block: receive or read+mcast the
block's in1, THEN wait for compute and write the block's output tiles, with a write barrier per subblock. The next
block's in1 cannot move until the previous block's last output tile has been computed, packed and written, so on a
batched matmul with one K block per batch (the trimul einsum: 128 batches of a 736^3 matmul, in0_block_w = Kt) the
DRAM read + mcast of every batch's in1 sits fully exposed between two compute phases.

The patch defers each block's output writes until after the NEXT block's in1 phase (one block of lookahead), and
waits for writes to have left L1 (`async_writes_flushed`) instead of their completion before freeing an output
subblock; the kernel's final `async_write_barrier` stays. Same tiles, same addresses, same order per block: the
output bytes cannot change. Deadlock-free for any CB depth: block t's in1 phase needs an in1 slot, which compute
frees after finishing block t-1 or earlier, whose output space was freed by writes of block t-2 or earlier, all of
which the writer issued in an earlier iteration. Not applied with fused all-gather / reduce-scatter or sparsity
(they synchronise per batch on completed writes): those keep the stock order through `if constexpr`.

Both stock files are matched by sha256, so a ttnn whose kernels differ gets no patch (and `build` raises).
"""
import hashlib, re

SENDER = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_sender_writer_padding.cpp"
RECEIVER = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow/reader_bmm_tile_layout_in1_receiver_writer_padding.cpp"
STOCK = {SENDER: "15e52a34bad837ccb99c1de0798df52f140890f39ad8cf0df12a722d8dfe6e63",
         RECEIVER: "102142710d5d9db1706605d13b40f626d3fb55d9a2616684a5989477056ef37d"}


def _check(rel, src):
    h = hashlib.sha256(src.encode()).hexdigest()
    if h != STOCK[rel]:
        raise RuntimeError(f"mm_pipe: {rel.rsplit('/', 1)[1]} is not the stock file this patch was written for ({h[:12]})")


def _cut_writer(src, start_anchor, end_anchor):
    """Split src around the writer section [start_anchor ... end_anchor) inside the block loop."""
    i = src.index(start_anchor)
    j = src.index(end_anchor, i)
    return src[:i], src[i:j], src[j:]


def _lambda(body, params):
    body = body.replace("noc.async_write_barrier();", "if constexpr (TB_PIPE) { noc.async_writes_flushed(); } else { noc.async_write_barrier(); }")
    return ("    auto tb_write_block = [&](" + params + ") {\n" + body + "    };\n")


_CALL = """#ifndef OUT_SHARDED
                    if constexpr (TB_PIPE) {
                        if (tb_pending) {
                            tb_write_block(tb_p_bh, tb_p_bw, tb_p_tile);
                        }
                        tb_pending = true;
                        tb_p_bh = bh;
                        tb_p_bw = bw;
                        tb_p_tile = out_tensor_current_w_dim_block_tile_id;
                    } else {
                        tb_write_block(bh, bw, out_tensor_current_w_dim_block_tile_id);
                    }
#endif
"""
_FLUSH = """#ifndef OUT_SHARDED
    if constexpr (TB_PIPE) {
        if (tb_pending) {
            tb_write_block(tb_p_bh, tb_p_bw, tb_p_tile);
        }
        noc.async_write_barrier();
    }
#endif
"""


def patch_sender(src):
    _check(SENDER, src)
    head, writer, tail = _cut_writer(
        src, "#ifndef OUT_SHARDED\n                    // WRITER\n",
        "                    in1_tensor_current_w_dim_block_tile_id += in1_tensor_next_w_dim_block_stride;\n")
    assert writer.rstrip().endswith("#endif"), writer[-200:]
    body = writer[len("#ifndef OUT_SHARDED\n"):writer.rstrip().rindex("#endif")]
    decl = ("#ifndef OUT_SHARDED\n"
            "    // tt-bio mm_pipe: a block's output is written after the next block's in1 transfer (see overlay).\n"
            "    constexpr bool TB_PIPE = !fuse_op_all_gather && !fuse_op_reduce_scatter && batchB == 0;\n"
            + _lambda(body, "uint32_t bh, uint32_t bw, uint32_t out_tensor_current_w_dim_block_tile_id")
            + "    bool tb_pending = false;\n    uint32_t tb_p_bh = 0, tb_p_bw = 0, tb_p_tile = 0;\n#endif\n")
    loop = "    for (uint32_t b = 0; b < batch; ++b) {\n        uint32_t in1_batch_tile_id = in1_tensor_start_tile_id;\n"
    assert head.count(loop) == 1
    head = head.replace(loop, decl + loop)
    end = "#if OUT_SHARDED\n    cb_out.wait_front(\n"
    assert tail.count(end) == 1
    tail = tail.replace(end, _FLUSH + end)
    return head + _CALL + tail


def patch_receiver(src):
    _check(RECEIVER, src)
    head, writer, tail = _cut_writer(
        src, "#ifndef OUT_SHARDED\n                // WRITER\n",
        "                out_tensor_current_w_dim_block_tile_id += out_tensor_next_w_dim_block_stride;\n")
    assert writer.rstrip().endswith("#endif"), writer[-200:]
    body = writer[len("#ifndef OUT_SHARDED\n"):writer.rstrip().rindex("#endif")]
    decl = ("#ifndef OUT_SHARDED\n"
            "    // tt-bio mm_pipe: a block's output is written after the next block's in1 transfer (see overlay).\n"
            "    constexpr bool TB_PIPE = !fuse_op_reduce_scatter;\n"
            + _lambda(body, "uint32_t bh, uint32_t bw, uint32_t out_tensor_current_w_dim_block_tile_id")
            + "    bool tb_pending = false;\n    uint32_t tb_p_bh = 0, tb_p_bw = 0, tb_p_tile = 0;\n#endif\n")
    loop = "    for (uint32_t b = 0; b < batch; ++b) {\n"
    assert head.count(loop) == 1
    head = head.replace(loop, decl + loop)
    end = "#if OUT_SHARDED\n    cb_out.wait_front(\n"
    assert tail.count(end) == 1
    tail = tail.replace(end, _FLUSH + end)
    return head + _CALL.replace("                    ", "                ") + tail
