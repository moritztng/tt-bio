"""Re-price the Pairformer block's non-arithmetic floor from the 2026-08-10 census.

ceiling-298aa.md §4 priced 217 non-matmul calls moving 5190 MB at the measured L1<->L1 clone
roof, 1152 GB/s counting read and write, giving 4.50 ms/block.  That construction is applied
here to the 192-call census in ledger_protenix-v2_320_qb1c3.json.
"""
import json
import sys

LEDGER = sys.argv[1] if len(sys.argv) > 1 else "perf/moonshot_k256/ledger_protenix-v2_320_qb1c3.json"
CLONE_ROOF_GBs = 1152.0  # ceiling-298aa.md §4, measured L1<->L1 clone, read+write counted

MATMUL_OPS = {
    "matmul", "linear", "minimal_matmul", "scaled_dot_product_attention",
}

d = json.load(open(LEDGER))
rows = d["rows"]


def is_matmul(r):
    op = r["op"]
    return op in MATMUL_OPS or "matmul" in op or "linear" in op


mm = [r for r in rows if is_matmul(r)]
na = [r for r in rows if not is_matmul(r)]

print("=== classification ===")
print("matmul-class rows      :", len(mm), " calls/block:", sum(r["n_per_block"] for r in mm),
      " ms/fold: %.1f" % sum(r["ms_per_fold"] for r in mm))
print("non-matmul rows        :", len(na), " calls/block:", sum(r["n_per_block"] for r in na),
      " ms/fold: %.1f" % sum(r["ms_per_fold"] for r in na))
print("matmul ops seen        :", sorted({r["op"] for r in mm}))

calls_per_fold = d["calls_per_fold"]

for label, sel in (("non-matmul", na), ("all ops", rows)):
    l1 = sum(r["l1_MB"] for r in sel)
    rd = sum(r["dram_read_MB"] for r in sel)
    wr = sum(r["dram_write_MB"] for r in sel)
    tot = l1 + rd + wr
    print()
    print("=== %s: bytes moved per block ===" % label)
    print("  l1_MB        %10.1f  -> %6.3f ms/block at %.0f GB/s" % (l1, l1 / CLONE_ROOF_GBs * 1e3 / 1e3 * 1e3 / 1e3, CLONE_ROOF_GBs))
    # MB / (GB/s) = MB / (1000 MB/ms)  -> ms
    def ms(mb):
        return mb / (CLONE_ROOF_GBs)  # GB/s == MB/ms
    print("  l1_MB        %10.1f MB -> %7.3f ms/block  (%7.2f s/fold)" % (l1, ms(l1), ms(l1) * calls_per_fold / 1e3))
    print("  dram_read_MB %10.1f MB -> %7.3f ms/block" % (rd, ms(rd)))
    print("  dram_write_MB%10.1f MB -> %7.3f ms/block" % (wr, ms(wr)))
    print("  TOTAL        %10.1f MB -> %7.3f ms/block  (%7.2f s/fold)" % (tot, ms(tot), ms(tot) * calls_per_fold / 1e3))

print()
print("=== non-matmul rows with zero l1_MB (i.e. priced only in DRAM) ===")
z = [r for r in na if r["l1_MB"] == 0.0]
print("  count %d of %d, ms/fold %.1f (%.1f%% of non-matmul)" % (
    len(z), len(na), sum(r["ms_per_fold"] for r in z),
    100 * sum(r["ms_per_fold"] for r in z) / max(1e-9, sum(r["ms_per_fold"] for r in na))))

print()
print("=== top 15 non-matmul rows ===")
for r in sorted(na, key=lambda r: -r["ms_per_fold"])[:15]:
    print("  %-30s %-22s n=%3d  %8.1f ms/fold  l1=%8.2f rd=%8.2f wr=%8.2f MB  %s %.1f%%" % (
        r["op"], r["site"], r["n_per_block"], r["ms_per_fold"],
        r["l1_MB"], r["dram_read_MB"], r["dram_write_MB"],
        r["binding_roof"], r["pct_of_binding_roof"] or 0.0))

print()
print("=== matmul-class rows ===")
for r in sorted(mm, key=lambda r: -r["ms_per_fold"]):
    print("  %-30s %-22s n=%3d  %8.1f ms/fold  %6.2f TF/s  %s %.1f%%  in=%s out=%s" % (
        r["op"], r["site"], r["n_per_block"], r["ms_per_fold"],
        r["achieved_TFLOPs"] or 0.0, r["binding_roof"], r["pct_of_binding_roof"] or 0.0,
        [i["shape"] for i in r["in"]], r["out"]["shape"]))
