"""Pass 6b: does the NO-GO survive the most fusion-favourable reading of the serial census?

Two known biases push the matmul side up, and both help fusion if removed:
  - the eight recovered rows were timed serially, with no dispatch overlap, so each is an upper
    bound on its own cost;
  - sites 2423 and 2434 carry the identical shape and differ 3.0x, so at least one is wrong, and
    the slow one is the one that hurts fusion.

So build the floor again with every matmul class priced at the FASTEST rate any class of that
exact shape reached in this census, rescale the whole attribution to the stage timer, and hand
the still-untimed `1659` to fusion for free. That is a lower bound on the matmul term and
therefore an upper bound on `f`. Then solve for what would have to be true to reach GO.
"""
import json

NEW = "perf/moonshot_k256/ledger_protenix-v2_320_qb1c3_serial.json"
CLONE_ROOF_GBs, STAGE_MS, BLOCKS, F_REQUIRED = 1152.0, 15400.0, 480.0, 1.677
MM = {"matmul", "linear", "minimal_matmul", "scaled_dot_product_attention"}

rows = json.load(open(NEW))["rows"]
is_mm = lambda r: r["op"] in MM
shape_sig = lambda r: (tuple(tuple(i["shape"]) for i in r["in"]),
                       tuple(r["out"]["shape"]) if r["out"] else None)

na = [r for r in rows if not is_mm(r)]
mm = [r for r in rows if is_mm(r)]
NA_FLOOR = sum(r["l1_MB"] + r["dram_read_MB"] + r["dram_write_MB"] for r in na) / CLONE_ROOF_GBs
na_ms = sum(r["ms_per_fold"] for r in na)
STAGE_BLOCK = STAGE_MS / BLOCKS

# fastest measured rate per exact shape signature
best = {}
for r in mm:
    if r["us_per_call"] > 0 and r["GFLOP_per_call"] > 0:
        tf = r["GFLOP_per_call"] / r["us_per_call"] * 1e3
        s = shape_sig(r)
        if tf > best.get(s, (0, None))[0]:
            best[s] = (tf, r["site"])

print("shape classes repriced downward (same shape, faster sibling measured in this census):")
mm_best = 0.0
for r in mm:
    s = shape_sig(r)
    if r["us_per_call"] == 0.0:
        continue                                  # `1659`: handed to fusion for free
    tf_best, src = best[s]
    tf_own = r["GFLOP_per_call"] / r["us_per_call"] * 1e3
    ms_own = r["ms_per_fold"]
    ms_best = ms_own * tf_own / tf_best
    mm_best += ms_best
    if ms_own - ms_best > 20:
        print("  %-16s %-22s n=%-3d %5.2f -> %5.2f TF/s (from %s)  %7.1f -> %7.1f ms/fold"
              % (r["op"], r["site"], r["n_per_block"], tf_own, tf_best, src, ms_own, ms_best))

mm_as_is = sum(r["ms_per_fold"] for r in mm)
print("\n  matmul term: %.0f ms/fold as measured -> %.0f ms/fold at best-of-shape (-%.0f%%)"
      % (mm_as_is, mm_best, 100 * (1 - mm_best / mm_as_is)))


def f_of(mm_ms, rescale=True):
    s = STAGE_MS / (mm_ms + na_ms) if rescale else 1.0
    return STAGE_BLOCK / (mm_ms * s / BLOCKS + NA_FLOOR)


print("\nfloor with the most fusion-favourable matmul term:")
for lab, v, rs in (("best-of-shape, rescaled to stage", mm_best, True),
                   ("best-of-shape, no rescale", mm_best, False),
                   ("as measured, rescaled to stage", mm_as_is, True)):
    f = f_of(v, rs)
    print("  %-34s f = %.3fx  %s" % (lab, f, "CLEARS" if f > F_REQUIRED else "SHORT"))

# what would have to be true
mm_block_max = STAGE_BLOCK / F_REQUIRED - NA_FLOOR
print("\nwhat GO would require:")
print("  matmul term must be below %.2f ms/block = %.0f ms/fold (%.1f%% of the stage)"
      % (mm_block_max, mm_block_max * BLOCKS, 100 * mm_block_max / STAGE_BLOCK))
for lab, v in (("as measured", mm_as_is), ("best-of-shape", mm_best)):
    s = STAGE_MS / (v + na_ms)
    print("  %-14s rescaled: %.0f ms/fold, %.0f%% above that bar"
          % (lab, v * s, 100 * (v * s / (mm_block_max * BLOCKS) - 1)))
print("  the non-arithmetic floor is byte-derived, not timed, so census noise cannot move it;")
print("  only the matmul/non-matmul split is noise-sensitive, and GO needs mm <= %.1f%% of stage."
      % (100 * mm_block_max / STAGE_BLOCK))
