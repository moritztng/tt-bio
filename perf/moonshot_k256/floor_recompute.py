"""Pass 5: re-price the non-arithmetic floor, and price the nine untimed census classes.

Step 1 of state doc 9d.  Two constructions of the Pairformer-block fusion floor:

  A  the ledger exactly as it stands, with the 4.50 ms/block non-arithmetic placeholder
     replaced by the byte-derived floor measured here;
  B  the same, after the nine classes whose standalone re-run threw are priced from measured
     rates for their own or an adjacent class, and the whole attribution is rescaled to the
     15.400 s stage timer it has to fit inside.

Everything in B is a projection and is labelled one.  A and B bracket the 1.677x block speedup
the megakernel needs, which is the point.
"""
import json

LEDGER = "perf/moonshot_k256/ledger_protenix-v2_320_qb1c3.json"
CLONE_ROOF_GBs = 1152.0     # ceiling-298aa.md §4, measured L1<->L1 clone, read+write counted
STAGE_MS = 15400.0          # state doc §8, instrumented `pairformer` stage, 298 aa
BLOCKS = 480.0              # 48 blocks x 10 recycles, protenix.py:2013/2241/2252
F_REQUIRED = 1.677          # state doc §9c: W=1.341x, P_512(fast)=0.630

d = json.load(open(LEDGER))
rows = d["rows"]
MM = {"matmul", "linear", "minimal_matmul", "scaled_dot_product_attention"}
is_mm = lambda r: r["op"] in MM


def gflop(shapes_in, shape_out, op):
    """2*M*K*N for a matmul/linear; SDPA is handled by its own recorded basis."""
    a, b = shapes_in[0], shapes_in[1]
    m = 1
    for x in a[:-1]:
        m *= x
    k, n = a[-1], b[-1]
    return 2.0 * m * k * n / 1e9


# ---- measured rates used to price the untimed classes -------------------------------------
# 52.81 TF/s: state doc §2, this exact [.,320,320,256]@[256,256] class at production config, qb1 c3.
# 34.46 TF/s: this ledger's own row at tenstorrent.py:2445, the adjacent K=256 transition
#             projection to DRAM ([1,32,320,1024]@[1024,256]).
RATE_FOR_UNTIMED = {
    ("minimal_matmul", "tenstorrent.py:1659"): 52.81,
    ("linear", "tenstorrent.py:798"): 52.81,
    ("linear", "tenstorrent.py:2423"): 34.46,
    ("linear", "tenstorrent.py:2434"): 34.46,
}

untimed = [r for r in rows if r["ms_per_fold"] == 0.0]
timed = [r for r in rows if r["ms_per_fold"] > 0.0]

print("=" * 78)
print("A. non-arithmetic floor, re-priced from the census (state doc 9d step 1)")
print("=" * 78)
na = [r for r in rows if not is_mm(r)]
mm = [r for r in rows if is_mm(r)]
ms = lambda mb: mb / CLONE_ROOF_GBs          # GB/s == MB/ms

l1 = sum(r["l1_MB"] for r in na)
rd = sum(r["dram_read_MB"] for r in na)
wr = sum(r["dram_write_MB"] for r in na)
tot = l1 + rd + wr
print("  non-matmul: %d classes, %d calls/block" % (len(na), sum(r["n_per_block"] for r in na)))
print("  l1 %.1f + dram_read %.1f + dram_write %.1f = %.1f MB/block" % (l1, rd, wr, tot))
print("  l1_MB alone            -> %.3f ms/block   (outside the [2,10] accept band)" % ms(l1))
print("  all bytes moved        -> %.3f ms/block   <-- replaces the 4.50 ms placeholder" % ms(tot))
zero_l1 = [r for r in na if r["l1_MB"] == 0.0]
print("  reason l1_MB alone fails: %d of %d non-matmul classes are DRAM-resident and record"
      % (len(zero_l1), len(na)))
print("  l1_MB = 0; ceiling-298aa.md §4's 5190 MB counted every byte those ops move.")
NA_FLOOR = ms(tot)

mm_ms = sum(r["ms_per_fold"] for r in mm)
na_ms = sum(r["ms_per_fold"] for r in na)
print()
print("  construction A, ledger as it stands:")
print("    matmul-class      %8.1f ms/fold = %6.2f ms/block (fusion does not remove it)" % (mm_ms, mm_ms / BLOCKS))
print("    non-matmul floor                    %6.2f ms/block" % NA_FLOOR)
fa = mm_ms / BLOCKS + NA_FLOOR
print("    fusion-only floor                   %6.2f ms/block -> f = %.3fx  (need %.3fx) %s"
      % (fa, (STAGE_MS / BLOCKS) / fa, F_REQUIRED,
         "CLEARS" if (STAGE_MS / BLOCKS) / fa > F_REQUIRED else "SHORT"))

print()
print("=" * 78)
print("B. the nine untimed classes, priced (projection)")
print("=" * 78)
print("  pf_block_ops.py:117 records a row with time 0.0 when the standalone re-run throws.")
print("  census_qb1c3.log: L1 OOM at 13107200 B and 52428800 B, plus static-CB clashes.")
add_mm = 0.0
for r in sorted(untimed, key=lambda r: r["site"]):
    key = (r["op"], r["site"])
    if key in RATE_FOR_UNTIMED:
        g = gflop([i["shape"] for i in r["in"]], r["out"]["shape"], r["op"])
        tf = g * r["n_per_block"] * BLOCKS / 1e3          # TFLOP per fold
        rate = RATE_FOR_UNTIMED[key]
        est = tf / rate * 1e3                              # ms per fold
        add_mm += est
        print("  %-16s %-22s n=%2d  %8.2f TFLOP/fold  @ %5.2f TF/s -> %7.1f ms/fold"
              % (r["op"], r["site"], r["n_per_block"], tf, rate, est))
    else:
        print("  %-16s %-22s n=%2d  non-matmul, not priced (bytes already counted above)"
              % (r["op"], r["site"], r["n_per_block"]))

mm_ms_b = mm_ms + add_mm
attributed = mm_ms_b + na_ms
scale = STAGE_MS / attributed
print()
print("  matmul-class          %8.1f -> %8.1f ms/fold  (+%.1f, +%.0f%%)" % (mm_ms, mm_ms_b, add_mm, 100 * add_mm / mm_ms))
print("  attributed total      %8.1f ms/fold against a %.0f ms stage = %.0f%% coverage" % (attributed, STAGE_MS, 100 / scale))
print("  rescale every term by %.4f so the attribution fits the stage it is inside" % scale)
mm_scaled = mm_ms_b * scale
na_scaled = na_ms * scale
print("    matmul-class      %8.1f ms/fold = %6.2f ms/block" % (mm_scaled, mm_scaled / BLOCKS))
print("    non-matmul        %8.1f ms/fold = %6.2f ms/block" % (na_scaled, na_scaled / BLOCKS))
print("    split             matmul %.1f%% / non-matmul %.1f%%   (ledger as-is: %.1f%% / %.1f%%)"
      % (100 * mm_scaled / STAGE_MS, 100 * na_scaled / STAGE_MS,
         100 * mm_ms / (mm_ms + na_ms), 100 * na_ms / (mm_ms + na_ms)))
fb = mm_scaled / BLOCKS + NA_FLOOR
print("    fusion-only floor                   %6.2f ms/block -> f = %.3fx  (need %.3fx) %s"
      % (fb, (STAGE_MS / BLOCKS) / fb, F_REQUIRED,
         "CLEARS" if (STAGE_MS / BLOCKS) / fb > F_REQUIRED else "SHORT"))

print()
print("=" * 78)
print("verdict on the go/no-go")
print("=" * 78)
print("  required block speedup            %.3fx" % F_REQUIRED)
print("  fusion-only floor, construction A %.3fx" % ((STAGE_MS / BLOCKS) / fa))
print("  fusion-only floor, construction B %.3fx" % ((STAGE_MS / BLOCKS) / fb))
print("  The two bracket the requirement. The nine untimed rows decide the sign.")
