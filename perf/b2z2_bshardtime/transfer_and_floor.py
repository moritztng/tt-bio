"""The block->fold transfer coefficient for `b_shard`, and the A/A floor bootstrapped at n=7.

Three things the orchestrator's 2026-09-13 amendment makes binding on this row.

1. `b2z2-trunk-fold-ab-bh` measured a block lever reaching the fold at transfer coefficient **1.0**.
   This row's lever reaches it at **a negative coefficient**, and saying so precisely is the point:
   the amendment's 1.0 was measured on a lever that DELETES READS, and `b_shard` does not delete
   reads -- it buys deleted compute with two added collectives. So the amendment's rule holds and
   its domain is now known: a read-deleting lever transfers at 1.0, a collective-buying one does not.

2. The A/A floor must be bootstrapped AT THE n THE RATIO IS QUOTED AT. The arm ratios here are
   ratios of a median of 7, so the null must be the ratio of two medians of 7 drawn from the
   lever-free population -- the 14 `mesh` folds, 7 at each end of a rep. A single split-half point
   estimate is one draw from that distribution and says nothing about its width.

3. Whether this row's term and `b2z2-reblock-axis2-gate`'s touch the same readers -- answered by
   `gate_reader_overlap.py`, not here.

The last section is the one that turns this row's "leading hypothesis" into an identity. The three
arms differ in how many collectives they take per block -- 0, 4 and 6 -- and share everything else,
so with the non-block remainder R fixed by the `mesh` arm, each sharded arm gives an INDEPENDENT
estimate of how much more a collective costs inside a fold than it does measured standalone on the
same mesh. Two estimates, one free parameter.

    python3 perf/b2z2_bshardtime/transfer_and_floor.py
"""

import json
import pathlib
import random
import statistics as st

HERE = pathlib.Path(__file__).resolve().parent
ab = json.loads((HERE / "fold_ab_n2_analysis.json").read_text())
raw = json.loads((HERE / "fold_ab_n2.json").read_text())
blk = json.loads((HERE / "curve_n2.json").read_text())

BOOT = 20000
random.seed(0)
warm = raw["folds"][3:]
reps = [warm[i * 4:(i + 1) * 4] for i in range(raw["reps"])]
mesh_trunk = [f["trunk_s"] for f in warm if f["arm"] == "mesh"]
assert len(mesh_trunk) == 2 * raw["reps"]
n = raw["reps"]

# --- 2. the A/A floor, bootstrapped at the n the ratio is quoted at ----------------------------
null = []
for _ in range(BOOT):
    a = [random.choice(mesh_trunk) for _ in range(n)]
    b = [random.choice(mesh_trunk) for _ in range(n)]
    null.append(st.median(b) / st.median(a))
null.sort()


def pct(v, p):
    return v[min(len(v) - 1, int(p * len(v)))]


lo, hi = pct(null, 0.025), pct(null, 0.975)
print(f"A/A floor on trunk_s, bootstrapped ratio of two medians of {n} from the {len(mesh_trunk)} "
      f"lever-free mesh folds, {BOOT} draws")
print(f"  95 % band  {lo:.5f}x - {hi:.5f}x      (median {st.median(null):.5f}x)")
print(f"  the split-half point estimate this row published was "
      f"{ab['by_statistic']['trunk_s']['aa_floor_matched']:.5f}x -- one draw from this band\n")

# --- the measured ratio, paired bootstrap over reps -------------------------------------------
pairs = [(next(f["trunk_s"] for f in r if f["arm"] == "shard"),
          next(f["trunk_s"] for f in r if f["arm"] == "bshard")) for r in reps]
boot = []
for _ in range(BOOT):
    d = [random.choice(pairs) for _ in range(n)]
    boot.append(st.median([s for s, _ in d]) / st.median([b for _, b in d]))
boot.sort()
r_lo, r_hi = pct(boot, 0.025), pct(boot, 0.975)
ratio = ab["by_statistic"]["trunk_s"]["bshard_vs_shard"]
print(f"bshard vs shard on trunk_s: {ratio:.5f}x, paired bootstrap 95 % {r_lo:.5f}x - {r_hi:.5f}x")
print(f"  clears the floor: {r_hi < lo}  (the whole interval is below the floor's lower edge)\n")

# --- 1. the transfer coefficient ---------------------------------------------------------------
b_shard_blk = blk["arms"]["shard"]["median_ms"], blk["arms"]["bshard"]["median_ms"]
blk_gain = (b_shard_blk[0] - b_shard_blk[1]) / b_shard_blk[0]
t = ab["by_statistic"]["trunk_s"]["median"]
trunk_gain = (t["shard"] - t["bshard"]) / t["shard"]
coef = trunk_gain / blk_gain
print(f"block   shard {b_shard_blk[0]:.3f} -> bshard {b_shard_blk[1]:.3f} ms   "
      f"{blk_gain*100:+.3f} %")
print(f"trunk   shard {t['shard']:.3f} -> bshard {t['bshard']:.3f} s    {trunk_gain*100:+.3f} %")
print(f"transfer coefficient {coef:+.3f}, against the +1.0 b2z2-trunk-fold-ab-bh measured for a "
      f"read-deleting lever\n")

# --- the collective's cost inside a fold, closed on two arms ----------------------------------
BLOCKS = raw["row_shard_calls_total"]["chain"] // 16      # 16 sharded folds: 2 cold + 14 warm
assert BLOCKS == 264, f"{BLOCKS} sharded blocks per fold, expected 264"
whole_ms = blk["arms"]["whole"]["median_ms"]
R = t["mesh"] - BLOCKS * whole_ms / 1e3                   # non-block trunk, fixed by the mesh arm
print(f"the trunk runs {BLOCKS} PairformerLayer blocks. With the mesh arm fixing the non-block "
      f"remainder at R = {R:.3f} s:")
print(f"  {'arm':7} {'gathers/blk':>11} {'block ms':>9} {'predicted':>10} {'measured':>9} "
      f"{'gap s':>7} {'per gather':>11}")
deltas = {}
# the fold calls the replicated arm `mesh`; the block harness calls it `whole`.
BLK_ARM = {"mesh": "whole", "shard": "shard", "bshard": "bshard"}
for arm, g in (("mesh", 0), ("shard", 4), ("bshard", 6)):
    b_ms = blk["arms"][BLK_ARM[arm]]["median_ms"]
    pred = R + BLOCKS * b_ms / 1e3
    gap = t[arm] - pred
    per = gap / (BLOCKS * g) * 1e3 if g else float("nan")
    if g:
        deltas[arm] = per
    print(f"  {arm:7} {g:>11d} {b_ms:>9.3f} {pred:>10.3f} {t[arm]:>9.3f} {gap:>7.3f} "
          f"{per:>10.3f} ms")
d_vals = list(deltas.values())
print(f"\ntwo arms, two independent estimates of the same quantity: {d_vals[0]:.3f} and "
      f"{d_vals[1]:.3f} ms, {abs(d_vals[0]/d_vals[1]-1)*100:.1f} % apart.")
gp, gb = (blk["gather"]["pair"]["median_us"] / 1e3, blk["gather"]["b"]["median_us"] / 1e3)
print(f"So a collective that measures {gp:.3f} ms (pair) / {gb:.3f} ms (b) standalone on this mesh "
      f"costs about {gp+st.mean(d_vals):.1f} / {gb+st.mean(d_vals):.1f} ms inside the fold -- "
      f"{(gp+st.mean(d_vals))/gp:.1f}x / {(gb+st.mean(d_vals))/gb:.1f}x.")

out = {"aa_floor_bootstrap": {"n": n, "draws": BOOT, "lo": lo, "hi": hi,
                              "median": st.median(null), "population": mesh_trunk},
       "ratio_bootstrap": {"point": ratio, "lo": r_lo, "hi": r_hi, "clears_floor": r_hi < lo},
       "transfer": {"block_gain_pct": blk_gain * 100, "trunk_gain_pct": trunk_gain * 100,
                    "coefficient": coef, "reference_read_deleting_lever": 1.0},
       "collective_cost": {"blocks_per_fold": BLOCKS, "remainder_s": R,
                           "per_gather_extra_ms": deltas,
                           "standalone_ms": {"pair": gp, "b": gb},
                           "in_fold_ms": {"pair": gp + st.mean(d_vals), "b": gb + st.mean(d_vals)}}}
(HERE / "transfer_and_floor.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {HERE / 'transfer_and_floor.json'}")
