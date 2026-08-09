"""Build a private kernel overlay tree instead of mutating the shared ttnn wheel.

tt-metal resolves a relative kernel path as CWD -> TT_METAL_KERNEL_PATH -> system dir ->
TT_METAL_HOME (tt_metal/impl/kernels/kernel.cpp resolve_path). So a tree that mirrors only the
files we want to change, exported as TT_METAL_KERNEL_PATH, overrides them for this process alone.
The wheel under site-packages stays byte-identical, which matters because seven other legs are
running matmuls on this host.

  python3 koverlay.py <mode>   -> prints the overlay dir to export as TT_METAL_KERNEL_PATH

modes: control (a #error, to prove the overlay is actually consulted)
       dualnoc  (issue alternate output tiles on NOC 1 from the same RISC)
       noc1     (issue every output tile on NOC 1)
"""
import os, re, shutil, sys

WHEEL = "/home/ttuser/.local/lib/python3.10/site-packages/ttnn"
REL = "ttnn/cpp/ttnn/operations/matmul/device/kernels/dataflow"
FILES = ["reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
         "reader_bmm_tile_layout_in1_sender_writer_padding.cpp"]
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "koverlay")

WRITE_LINE = "noc_async_write_tile(out_tensor_tile_id, s, l1_read_addr);"
BARRIER_LINE = "noc_async_write_barrier();"
POP_LINE = "cb_pop_front(cb_id_out0, out_subblock_tile_count);"
DEFS = ["constexpr uint32_t W3_TRID_A = 4;", "constexpr uint32_t W3_TRID_B = 5;"]


def nocsel(mode):
    if mode == "noc1":
        return "(uint8_t)1"
    if mode == "dualnoc":
        return "(uint8_t)(w & 1)"
    m = re.fullmatch(r"split(\d+)", mode)
    if m:
        return "(uint8_t)(w >= %s)" % m.group(1)
    raise SystemExit("bad mode " + mode)


def transform(src, mode):
    lines = src.split("\n")
    if mode == "control":
        return "#error W3_OVERLAY_ACTIVE\n" + src
    wi = [i for i, l in enumerate(lines) if l.strip() == WRITE_LINE]
    assert len(wi) == 1, wi
    pi = [i for i, l in enumerate(lines) if l.strip() == POP_LINE]
    assert len(pi) == 1, pi
    bi = max(i for i, l in enumerate(lines[:pi[0]]) if l.strip() == BARRIER_LINE)
    ki = [i for i, l in enumerate(lines) if l.strip() == "void kernel_main() {"]
    assert len(ki) == 1
    ind = lambda i: re.match(r"\s*", lines[i]).group(0)
    sel = nocsel(mode)
    lines[wi[0]] = (
        ind(wi[0]) + "{ const uint8_t w3n = " + sel + ";"
        " noc_async_write_one_packet_with_trid(l1_read_addr,"
        " s.get_noc_addr(out_tensor_tile_id, 0, w3n),"
        " output_single_tile_size_bytes, w3n ? W3_TRID_B : W3_TRID_A, (uint8_t)0, w3n); }")
    lines[bi] = (ind(bi) + "noc_async_write_barrier_with_trid(W3_TRID_A, 0);\n"
                 + ind(bi) + "noc_async_write_barrier_with_trid(W3_TRID_B, 1);")
    lines[ki[0]] = lines[ki[0]] + "\n" + "\n".join("    " + d for d in DEFS)
    return "\n".join(lines)


mode = sys.argv[1]
out = os.path.join(ROOT, mode)
if mode == "stock":
    print("")            # no overlay at all
    raise SystemExit(0)
d = os.path.join(out, REL)
os.makedirs(d, exist_ok=True)
for f in FILES:
    src = open(os.path.join(WHEEL, REL, f)).read()
    assert "W3_TRID_A" not in src, "wheel source is NOT pristine: " + f
    open(os.path.join(d, f), "w").write(transform(src, mode))
print(out)
