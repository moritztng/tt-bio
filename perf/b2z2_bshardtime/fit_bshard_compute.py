"""The constant term of compute(N) with the `b` role split, against the 28.679 ms it has to beat.

`b2z2-trunk-shard-scale-wh` fitted `compute(N) = block_wall(N) - 4 x G_pair(N)` over N = 2, 4, 8 and
got `28.679 ms + 58.916 ms/N` at R2 0.9964. That constant is the campaign's 33.4 % replicated floor
and its 2.299x block cap. This file fits the same quantity with `b_shard` on, which needs SIX
collectives subtracted rather than four:

    compute_shard(N)  = block_wall(N) - 4 x G_pair(N)
    compute_bshard(N) = block_wall(N) - 4 x G_pair(N) - 2 x G_b(N)

Both arms come from `bshard_compute_curve.py`, which times them round-robin in one process on one
mesh and measures both collectives there too, value-checked. Two widths only -- cards 20-23 are
`b2z2-reblock-axis2-gate`'s and 16-19 is the tray this row holds -- so the fit is a two-point one
and the SHARD arm is refitted on the same two points so the comparison is like-for-like. The
three-point 28.679 is then moved by the measured DIFFERENCE of the two constants, which is the one
quantity a two-point fit estimates well: it is the compute the `b` role stops replicating, and
subtracting the link from both arms leaves the fit no freedom to put it anywhere else.

    python3 perf/b2z2_bshardtime/fit_bshard_compute.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
SCALE = HERE.parents[0] / "b2z2_shardscale"

PARENT_A_3PT = 28.679          # b2z2-trunk-shard-scale-wh, compute(N) over N=2,4,8, R2 0.9964
PARENT_B_3PT = 58.916
PARENT_BASE = 85.740           # its one-chip block
PARENT_CAP = 2.299             # base / (A + 4 x G_pair(8)); ceiling_v3.SHARD_CAP_MEASURED_LINK
CELL_S = 20.113                # published Boltz-2 512 aa Blackhole cell
PAIRFORMER_S = 10.22           # PairformerLayer device spans inside it, CONTEXT 2-CORRECTION


def load(n):
    return json.loads((HERE / f"curve_n{n}.json").read_text())


runs = {n: load(n) for n in (2, 4)}
base = sum(r["arms"]["whole"]["median_ms"] for r in runs.values()) / len(runs)

# The N=8 pair gather is the parent's, from the same instrument on the same box. There is no N=8
# `b` gather and this row could not take one, so it is carried forward from N=4 by the growth the
# PAIR gather shows between those widths. Labelled an extrapolation everywhere it is used.
GP8 = next(p for p in json.loads((SCALE / "b2z2_gathercurve_8.json").read_text())["pair_shape"]
           if p["S"] == 512)["median_us"] / 1e3


def fit2(y2, y4):
    """a + b/N through two widths: b = 4(y2 - y4), a = 2*y4 - y2."""
    return 2 * y4 - y2, 4 * (y2 - y4)


rows, comp = [], {}
print(f"one-chip block (mean of the two meshes): {base:.3f} ms")
print()
print(f"{'N':>2} {'G_pair':>8} {'G_b':>8} | {'whole':>8} {'shard':>8} {'bshard':>8} | "
      f"{'c_shard':>8} {'c_bshd':>8} | {'load':>6}")
for n in sorted(runs):
    r = runs[n]
    gp, gb = r["gather"]["pair"]["median_us"] / 1e3, r["gather"]["b"]["median_us"] / 1e3
    a = {k: r["arms"][k]["median_ms"] for k in ("whole", "shard", "bshard")}
    cs = a["shard"] - 4 * gp
    cb = a["bshard"] - 4 * gp - 2 * gb
    comp[n] = {"G_pair_ms": gp, "G_b_ms": gb, "link_shard_ms": 4 * gp,
               "link_bshard_ms": 4 * gp + 2 * gb, "compute_shard_ms": cs, "compute_bshard_ms": cb,
               **a}
    load = max(r["loadavg_per_rep"])
    print(f"{n:>2} {gp:>8.3f} {gb:>8.3f} | {a['whole']:>8.3f} {a['shard']:>8.3f} "
          f"{a['bshard']:>8.3f} | {cs:>8.3f} {cb:>8.3f} | {load:>6.1f}")
    rows.append(n)

A_s, B_s = fit2(comp[2]["compute_shard_ms"], comp[4]["compute_shard_ms"])
A_b, B_b = fit2(comp[2]["compute_bshard_ms"], comp[4]["compute_bshard_ms"])
dA = A_s - A_b
print()
print(f"compute_shard(N)  = {A_s:7.3f} ms + {B_s:7.3f} ms/N    "
      f"un-shardable {100*A_s/base:5.1f} % of the block")
print(f"compute_bshard(N) = {A_b:7.3f} ms + {B_b:7.3f} ms/N    "
      f"un-shardable {100*A_b/base:5.1f} % of the block")
print(f"\nthe constant falls by {dA:.3f} ms. The `b` role was attributed at 10.18 ms.")

# The three-point fit is the one the campaign quotes. Move it by the measured difference: that
# difference is a COMPUTE term, independent of how many widths support the fit, while the absolute
# constant of a two-point fit is not.
A_b_3pt = PARENT_A_3PT - dA
print(f"on the parent's three-point basis: {PARENT_A_3PT:.3f} - {dA:.3f} = {A_b_3pt:.3f} ms, "
      f"{100*A_b_3pt/PARENT_BASE:.1f} % of the block (was {100*PARENT_A_3PT/PARENT_BASE:.1f} %)")

# Caps. The saturated link is what a wide mesh pays forever: 4 pair gathers, plus 2 `b` gathers
# for the bshard arm. G_b(8) is extrapolated from G_b(4) by the pair gather's own 4->8 growth.
gp4, gb4 = comp[4]["G_pair_ms"], comp[4]["G_b_ms"]
gb8 = gb4 * (GP8 / gp4)
link_s8, link_b8 = 4 * GP8, 4 * GP8 + 2 * gb8
cap_s = PARENT_BASE / (PARENT_A_3PT + link_s8)
cap_b = PARENT_BASE / (A_b_3pt + link_b8)
print()
print(f"saturated link at N=8: shard {link_s8:.3f} ms, bshard {link_b8:.3f} ms "
      f"(G_b(8) = {gb8:.3f} ms EXTRAPOLATED from G_b(4) by the pair gather's 4->8 growth)")
print(f"block cap, measured link:  shard {cap_s:.3f}x   bshard {cap_b:.3f}x   "
      f"({cap_b/cap_s:.4f}x)")
print(f"block cap, free link:      shard {PARENT_BASE/PARENT_A_3PT:.3f}x   "
      f"bshard {PARENT_BASE/A_b_3pt:.3f}x")
print(f"\nceiling_v3.SHARD_CAP_MEASURED_LINK = {PARENT_CAP} becomes {cap_b:.3f}")

print()
print("projected onto the published cell -- WH block ratios on BH seconds, an upper bound:")
for label, cap in (("shard, any N, measured link", cap_s),
                   ("bshard, any N, measured link", cap_b),
                   ("shard, any N, free link", PARENT_BASE / PARENT_A_3PT),
                   ("bshard, any N, free link", PARENT_BASE / A_b_3pt)):
    trunk = PAIRFORMER_S / cap
    fold = CELL_S - PAIRFORMER_S + trunk
    print(f"  {label:30s} block {cap:6.3f}x  trunk {trunk:6.3f} s  fold {fold:7.3f} s  "
          f"{CELL_S/fold:.4f}x")

out = {
    "base_ms": base, "by_width": comp,
    "fit": {"shard": {"A_ms": A_s, "B_ms": B_s, "points": 2},
            "bshard": {"A_ms": A_b, "B_ms": B_b, "points": 2}},
    "delta_A_ms": dA, "b_role_attributed_ms": 10.18,
    "parent_three_point": {"A_ms": PARENT_A_3PT, "B_ms": PARENT_B_3PT, "base_ms": PARENT_BASE},
    "A_bshard_three_point_basis_ms": A_b_3pt,
    "replicated_pct": {"shard_2pt": 100 * A_s / base, "bshard_2pt": 100 * A_b / base,
                       "shard_3pt": 100 * PARENT_A_3PT / PARENT_BASE,
                       "bshard_3pt_basis": 100 * A_b_3pt / PARENT_BASE},
    "link_saturated_ms": {"shard": link_s8, "bshard": link_b8,
                          "G_pair_8_ms": GP8, "G_b_8_ms_extrapolated": gb8},
    "cap_measured_link": {"shard": cap_s, "bshard": cap_b, "ratio": cap_b / cap_s},
    "cap_free_link": {"shard": PARENT_BASE / PARENT_A_3PT, "bshard": PARENT_BASE / A_b_3pt},
    "block_ratios_same_mesh": {
        str(n): {"shard": comp[n]["whole"] / comp[n]["shard"],
                 "bshard": comp[n]["whole"] / comp[n]["bshard"],
                 "bshard_vs_shard": comp[n]["shard"] / comp[n]["bshard"]} for n in rows},
}
(HERE / "fit_bshard_compute.json").write_text(json.dumps(out, indent=1))
print(f"\nwrote {HERE / 'fit_bshard_compute.json'}")
