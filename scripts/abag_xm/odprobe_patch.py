#!/usr/bin/env python3
"""Probe-only instrumentation for the opendde-abag 9j4c L1 circular-buffer clash.

Applied to a throwaway copy of the engine tree on the Galaxy, never to `main`. It answers one
question: which live L1 buffer sits at address 479232 when TriangleAttention's chunked qkv
`minimal_matmul` tries to place its static circular buffers.

Adds `_odprobe_l1(tag)`, which prints the L1 memory view and dumps the allocator's buffer table,
and calls it on entry to TriangleAttention.__call__ and again immediately before the first chunked
`minimal_matmul`. Everything is gated on TT_BIO_ODPROBE_L1LOG=1, so an unset environment runs the
stock code path.

Usage:  python3 odprobe_patch.py <path-to-tt_bio/tenstorrent.py>
"""
import sys

HELPER = '''

_ODPROBE_DUMPED = set()


def _odprobe_l1(tag):
    """Probe-only: print the L1 memory view and dump the buffer table once per tag."""
    if os.environ.get("TT_BIO_ODPROBE_L1LOG") != "1":
        return
    dev = get_device()
    try:
        mv = ttnn.get_memory_view(dev, ttnn.BufferType.L1)
        print("ODPROBE [" + tag + "] L1 " + str(mv), flush=True)
    except Exception as exc:
        print("ODPROBE [" + tag + "] view-failed " + type(exc).__name__ + ": " + str(exc), flush=True)
    if tag not in _ODPROBE_DUMPED:
        _ODPROBE_DUMPED.add(tag)
        pre = os.environ.get("TT_BIO_ODPROBE_DUMP", "")
        if pre:
            try:
                ttnn.dump_device_memory_state(dev, pre + tag.replace(" ", "_") + "_")
                print("ODPROBE [" + tag + "] dumped", flush=True)
            except Exception as exc:
                print("ODPROBE [" + tag + "] dump-failed " + type(exc).__name__ + ": " + str(exc),
                      flush=True)


def _open_and_init_device(trace_region_size):'''

ANCHOR_HELPER = "\n\ndef _open_and_init_device(trace_region_size):"

ANCHOR_ENTRY = """        x = ttnn.reshape(x, tuple(x.shape)[1:])
        S = x.shape[0]
        need_chunk = S > SEQ_LEN_MORE_CHUNKING and (self.affinity or not _FAST_MODE or _IS_SMALL_GRID)"""

ENTRY_ADD = """
        _odprobe_l1("tri_att " + ("end" if self.ending else "start") + " entry S" + str(S))"""

ANCHOR_MM = """                x_chunk = normed_rows(s, end)
                qkv_chunk = ttnn.experimental.minimal_matmul("""

MM_ADD = """                x_chunk = normed_rows(s, end)
                _odprobe_l1("tri_att " + ("end" if self.ending else "start") + " pre-qkv s" + str(s))
                qkv_chunk = ttnn.experimental.minimal_matmul("""

# The chunked row loop stores its block into `parts`; make its L1 destination switchable so the
# same tree can test whether that call site is the holder.
ANCHOR_DEF = """        def gate_and_project(o_in: ttnn.Tensor, g_in: ttnn.Tensor) -> ttnn.Tensor:
            o_in = ttnn.multiply_(o_in, g_in, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g_in)
            x_out = _pair_proj_linear(
                o_in, self.o_weight, self.compute_kernel_config, _dtype(), l1_out=True
            )"""

DEF_NEW = """        def gate_and_project(o_in, g_in, l1_out=True):
            o_in = ttnn.multiply_(o_in, g_in, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
            ttnn.deallocate(g_in)
            x_out = _pair_proj_linear(
                o_in, self.o_weight, self.compute_kernel_config, _dtype(), l1_out=l1_out
            )"""

ANCHOR_CALL = "                _acc_append(parts, gate_and_project(o_chunk, g_chunk), host_acc)"

CALL_NEW = """                _acc_append(parts, gate_and_project(
                    o_chunk, g_chunk,
                    l1_out=os.environ.get("TT_BIO_ODPROBE_L1OUT", "1") == "1"), host_acc)"""


def main(path):
    src = open(path).read()
    for anchor, repl in (
        (ANCHOR_HELPER, HELPER),
        (ANCHOR_ENTRY, ANCHOR_ENTRY + ENTRY_ADD),
        (ANCHOR_MM, MM_ADD),
        (ANCHOR_DEF, DEF_NEW),
        (ANCHOR_CALL, CALL_NEW),
    ):
        n = src.count(anchor)
        if n != 1:
            raise SystemExit("anchor matched %d times, expected 1:\n%s" % (n, anchor[:120]))
        src = src.replace(anchor, repl, 1)
    open(path, "w").write(src)
    compile(src, path, "exec")
    print("PATCHED + SYNTAX OK")


if __name__ == "__main__":
    main(sys.argv[1])
