#!/usr/bin/env python3
"""Read the block-timing sessions and price the lever at the block, then at the fold.

Two things the harness summary cannot do on its own:

1. **Drop the first hoist fold from the steady-state median.** `DiffusionTransformer._cond_weights()`
   concatenates the 24 layers' conditioning projections once per PROCESS and caches them on the
   module, and the build lands inside the walled block of whichever fold calls it first. It is a
   one-time cost, not a per-fold one, so it is reported separately rather than averaged in.
2. **Pair the arms rep by rep.** The base arm runs at two positions of every rep, so each rep gives
   one paired delta (hoist against the mean of its own rep's two bases) AND one A/A delta (the two
   bases against each other). The A/A deltas are the session's own noise floor for the same
   quantity, in the same arm list.
"""
import json
import statistics as st
import sys
from pathlib import Path


def session(path):
    d = json.loads(Path(path).read_text())
    warm = [r for r in d["runs"] if not r["cold"]]
    reps = sorted({r["rep"] for r in warm})
    rows = []
    for rep in reps:
        rr = [r for r in warm if r["rep"] == rep]
        b = [r for r in rr if r["arm"] == "base"]
        h = [r for r in rr if r["arm"] == "hoist"]
        if len(b) != 2 or len(h) != 1:
            continue
        bb = st.mean(r["block_s"] for r in b)
        rows.append({
            "rep": rep,
            "block_base": bb,
            "block_hoist": h[0]["block_s"],
            "block_delta": bb - h[0]["block_s"],
            "block_ratio": bb / h[0]["block_s"],
            "aa_block_delta": b[0]["block_s"] - b[1]["block_s"],
            "fold_base": st.mean(r["fold_s"] for r in b),
            "fold_hoist": h[0]["fold_s"],
            "fold_delta": st.mean(r["fold_s"] for r in b) - h[0]["fold_s"],
            "aa_fold_delta": b[0]["fold_s"] - b[1]["fold_s"],
        })
    return d, warm, rows


def report(path):
    d, warm, rows = session(path)
    print(f"=== {path}")
    print(f"host={d['env']['host']} commit={d['env']['commit'][:9]} "
          f"visible={d['env']['tt_visible_devices']} load_at_start={d['env']['loadavg'][0]}")
    print(f"model_load_s={d.get('model_load_s')}")
    wit = {"block_n": sorted({r["block_n"] for r in warm}),
           "atom_block_n": sorted({r.get("atom_block_n") for r in warm}),
           "hoist_norms_base": sorted({r["hoist_norms"] for r in warm if r["arm"] == "base"}),
           "hoist_norms_hoist": sorted({r["hoist_norms"] for r in warm if r["arm"] == "hoist"})}
    print("WITNESS", json.dumps(wit))
    print(f"{'rep':>4} {'blk_base':>9} {'blk_hoist':>9} {'blk_d':>8} {'blk_x':>8} "
          f"{'AA_blk_d':>9} {'fold_base':>9} {'fold_hoist':>10} {'fold_d':>8} {'AA_fold_d':>9}")
    for r in rows:
        print(f"{r['rep']:>4} {r['block_base']:9.4f} {r['block_hoist']:9.4f} "
              f"{r['block_delta']:+8.4f} {r['block_ratio']:8.5f} {r['aa_block_delta']:+9.4f} "
              f"{r['fold_base']:9.3f} {r['fold_hoist']:10.3f} {r['fold_delta']:+8.3f} "
              f"{r['aa_fold_delta']:+9.3f}")

    first_hoist = min((r for r in warm if r["arm"] == "hoist"), key=lambda r: (r["rep"], r["pos"]))
    later = [r["block_s"] for r in warm
             if r["arm"] == "hoist" and (r["rep"], r["pos"]) != (first_hoist["rep"], first_hoist["pos"])]
    build = None
    if later:
        build = first_hoist["block_s"] - st.median(later)
        print(f"\nfirst hoist fold (rep {first_hoist['rep']}) block {first_hoist['block_s']:.4f}s vs "
              f"median of the later hoist folds {st.median(later):.4f}s "
              f"-> one-time _cond_weights() build {build:+.4f}s")

    ss = [r for r in rows if r["rep"] != first_hoist["rep"]]
    for label, sel in (("all reps", rows), ("steady state (build fold dropped)", ss)):
        if not sel:
            continue
        bd = [r["block_delta"] for r in sel]
        aa = [abs(r["aa_block_delta"]) for r in sel]
        fd = [r["fold_delta"] for r in sel]
        aaf = [abs(r["aa_fold_delta"]) for r in sel]
        print(f"\n-- {label}, n={len(sel)} reps")
        print(f"   block delta   median {st.median(bd):+.4f}s  range "
              f"[{min(bd):+.4f}, {max(bd):+.4f}]")
        print(f"   block A/A     median |delta| {st.median(aa):.4f}s  max {max(aa):.4f}s")
        print(f"   block ratio   median {st.median(r['block_ratio'] for r in sel):.5f}x")
        print(f"   fold  delta   median {st.median(fd):+.3f}s  range [{min(fd):+.3f}, {max(fd):+.3f}]")
        print(f"   fold  A/A     median |delta| {st.median(aaf):.3f}s  max {max(aaf):.3f}s")
        base_block = st.median(r["block_base"] for r in sel)
        base_fold = st.median(r["fold_base"] for r in sel)
        print(f"   COMPOSED: the walled block is {base_block:.4f}s of a {base_fold:.3f}s fold "
              f"= {100 * base_block / base_fold:.1f}% of it, so a block delta of "
              f"{st.median(bd):+.4f}s is at most {st.median(bd):+.4f}s of fold "
              f"({100 * st.median(bd) / base_fold:+.2f}% of the fold)")
    return rows, build


if __name__ == "__main__":
    for p in sys.argv[1:]:
        report(p)
        print()
