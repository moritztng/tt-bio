"""Pass 6: recompute the Pairformer-block fusion-only floor on the serial census, and put an
error bar on it by differencing the two census runs.

state doc 10e step 2. Three things this prints that floor_recompute.py could not:

  1. the go/no-go on the serial ledger, where 8 of the 9 previously untimed classes now have a
     measured time and only `1659` is left;
  2. `1659` priced from site `798`, which carries the identical shape class and was measured in
     this same session under this same load, rather than from the standalone 52.81 TF/s of §2;
  3. a reproducibility band, taken over the classes both census runs timed, because a decision
     that turns on a 52/48 split cannot be read off a census whose rows move by more than that
     between runs.
"""
import json
import statistics
import sys

OLD = "perf/moonshot_k256/ledger_protenix-v2_320_qb1c3.json"
NEW = "perf/moonshot_k256/ledger_protenix-v2_320_qb1c3_serial.json"
CLONE_ROOF_GBs = 1152.0     # ceiling-298aa.md §4, measured L1<->L1 clone, read+write counted
STAGE_MS = 15400.0          # state doc §8, instrumented `pairformer` stage, 298 aa
BLOCKS = 480.0              # 48 blocks x 10 recycles
F_REQUIRED = 1.677          # state doc §9c: W = 1.341x, P_512(fast) = 0.630

MM = {"matmul", "linear", "minimal_matmul", "scaled_dot_product_attention"}
is_mm = lambda r: r["op"] in MM
key = lambda r: (r["op"], r["site"], r["n_per_block"])
ms = lambda mb: mb / CLONE_ROOF_GBs          # GB/s == MB/ms


def load(p):
    return json.load(open(p))["rows"]


new, old = load(NEW), load(OLD)
newx = {key(r): r for r in new}
oldx = {key(r): r for r in old}

print("=" * 78)
print("0. reproducibility, over the classes both runs timed")
print("=" * 78)
both = [(k, oldx[k], newx[k]) for k in newx
        if k in oldx and oldx[k]["us_per_call"] > 0 and newx[k]["us_per_call"] > 0]
ratios = sorted((n["us_per_call"] / o["us_per_call"], k) for k, o, n in both)
r_only = [r for r, _ in ratios]
print("  %d classes timed in both runs" % len(both))
print("  new/old per-call ratio: median %.3f  p10 %.3f  p90 %.3f  min %.3f  max %.3f"
      % (statistics.median(r_only), r_only[int(0.1 * len(r_only))], r_only[int(0.9 * len(r_only))],
         r_only[0], r_only[-1]))
within = sum(1 for r in r_only if 0.9 <= r <= 1.1)
print("  %d of %d (%.0f%%) agree within 10%%" % (within, len(r_only), 100 * within / len(r_only)))
print("  worst movers, both runs timed, neither serial:")
for rr, k in ratios[:3] + ratios[-3:]:
    print("    %-16s %-22s n=%-3d %8.1f -> %8.1f us  x%.2f"
          % (k[0], k[1], k[2], oldx[k]["us_per_call"], newx[k]["us_per_call"], rr))

# ---- price the one class still untimed ---------------------------------------------------
untimed = [r for r in new if r["ms_per_fold"] == 0.0]
anchors = [r for r in new if r["us_per_call"] > 0
           and abs(r["GFLOP_per_call"] - 13.42177) < 1e-3]
print()
print("=" * 78)
print("1. the one class still untimed, and the anchor for it")
print("=" * 78)
for r in untimed:
    print("  UNTIMED %-16s %-22s n=%d  %.3f GFLOP/call" % (r["op"], r["site"], r["n_per_block"],
                                                           r["GFLOP_per_call"]))
print("  same-shape classes measured in THIS run (%.3f GFLOP/call):" % 13.42177)
rates = []
for r in anchors:
    tf = r["GFLOP_per_call"] / r["us_per_call"] * 1e3
    rates.append(tf)
    print("    %-16s %-22s n=%-3d %8.1f us -> %5.2f TF/s" % (r["op"], r["site"], r["n_per_block"],
                                                             r["us_per_call"], tf))
ANCHOR = statistics.median(rates)
print("  anchor rate (median of the above, same session, same load): %.2f TF/s" % ANCHOR)
print("  state doc §2 standalone rate for this class, for contrast:   52.81 TF/s")

# ---- the floor ---------------------------------------------------------------------------
na = [r for r in new if not is_mm(r)]
mm = [r for r in new if is_mm(r)]
NA_FLOOR = ms(sum(r["l1_MB"] + r["dram_read_MB"] + r["dram_write_MB"] for r in na))
mm_ms = sum(r["ms_per_fold"] for r in mm)
na_ms = sum(r["ms_per_fold"] for r in na)

print()
print("=" * 78)
print("2. the fusion-only floor on the serial census")
print("=" * 78)
print("  non-arithmetic floor from bytes moved: %.3f ms/block" % NA_FLOOR)


def report(label, mm_ms_in, rescale):
    attributed = mm_ms_in + na_ms
    s = STAGE_MS / attributed if rescale else 1.0
    mm_b = mm_ms_in * s / BLOCKS
    floor = mm_b + NA_FLOOR
    f = (STAGE_MS / BLOCKS) / floor
    print("  %-42s matmul %6.2f + nonarith %4.2f = %6.2f ms/block -> f = %.3fx  %s"
          % (label, mm_b, NA_FLOOR, floor, f, "CLEARS" if f > F_REQUIRED else "SHORT"))
    print("      %sattribution %.0f ms/fold = %.0f%% of the %.0f ms stage, split %.1f%% mm / %.1f%% na"
          % ("rescaled by %.4f, " % s if rescale else "", attributed * s, 100 * attributed / STAGE_MS,
             STAGE_MS, 100 * mm_ms_in * s / (attributed * s), 100 * na_ms * s / (attributed * s)))
    return f


tf_1659 = sum(r["GFLOP_per_call"] * r["n_per_block"] * BLOCKS for r in untimed) / 1e3
add = tf_1659 / ANCHOR * 1e3
print("  `1659` priced: %.2f TFLOP/fold @ %.2f TF/s = %.0f ms/fold" % (tf_1659, ANCHOR, add))
print()
f_a = report("A  serial ledger as it stands", mm_ms, False)
f_b = report("B  + `1659` priced, all rescaled to stage", mm_ms + add, True)
f_c = report("C  + `1659` priced, no rescale", mm_ms + add, False)

print()
print("=" * 78)
print("verdict")
print("=" * 78)
print("  required block speedup       %.3fx" % F_REQUIRED)
for lab, f in (("A", f_a), ("B", f_b), ("C", f_c)):
    print("  construction %s               %.3fx  %s" % (lab, f, "CLEARS" if f > F_REQUIRED else "SHORT"))
print("  10e step 2 band: NO-GO if the answer lands in [1.6, 1.75] -- a megakernel that has to")
print("  reach 96%% of its own theoretical floor is not a program worth starting.")
