"""Patch the two matmul writer kernels so the output drain issues on both NOCs.

The stock writer issues every output tile with noc_async_write_tile, which on BRISC always goes out
on NOC 0. This rewrites the issue to noc_async_write_one_packet_with_trid so the NOC can be chosen
per tile, and replaces the per-subblock noc_async_write_barrier with a per-trid barrier on each NOC.

Why the per-trid barrier: BRISC's software write counters are only initialised for its own NOC
(brisck.cc calls noc_local_state_init(NOC_INDEX)), so noc_async_write_barrier(1) from BRISC compares
against an uninitialised counter and hangs. ncrisc_noc_nonposted_write_with_transaction_id_flushed
reads NIU_MST_REQS_OUTSTANDING_ID(trid) straight out of hardware, which is exact and does not care
what the other RISC is doing.

Command buffer 0 is BRISC_WR_CMD_BUF and NCRISC_WR_CMD_BUF both (on Blackhole the two RISCs are kept
apart by NOC, not by command buffer), and noc_init sets its source coordinate on both NOCs, so
issuing from BRISC on NOC 1 through cmd buf 0 is configured hardware. It collides only if the in0
reader on NCRISC issues a large write on NOC 1 at the same time; it only reads and does semaphore
ops, which use cmd bufs 1, 2 and 3.

Usage: python3 patch_dualnoc.py {stock|dualnoc|noc1|split<N>}
"""
import os, re, shutil, sys

K = ("/home/ttuser/.local/lib/python3.10/site-packages/ttnn/ttnn/cpp/ttnn/operations/matmul/"
     "device/kernels/dataflow")
FILES = ["reader_bmm_tile_layout_in1_receiver_writer_padding.cpp",
         "reader_bmm_tile_layout_in1_sender_writer_padding.cpp"]

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
    if m:                      # first N tiles of the row on noc0, rest on noc1
        return "(uint8_t)(w >= %s)" % m.group(1)
    raise SystemExit("bad mode " + mode)


def patch(path, mode):
    backup = path + ".w3_backup"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
    lines = open(backup).read().split("\n")
    if mode == "stock":
        shutil.copy2(backup, path)
        return "restored"

    wi = [i for i, l in enumerate(lines) if l.strip() == WRITE_LINE]
    assert len(wi) == 1, (path, wi)
    pi = [i for i, l in enumerate(lines) if l.strip() == POP_LINE]
    assert len(pi) == 1, (path, pi)
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
    open(path, "w").write("\n".join(lines))
    return "patched"


mode = sys.argv[1]
for f in FILES:
    print(f, patch(os.path.join(K, f), mode))
print("MODE " + mode)
