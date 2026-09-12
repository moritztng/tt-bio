"""The block-wall fit for both shards, and the cap each one implies, from the committed JSON.

Two points, not three: cards 24-27 belong to `b2z2-atom-axis-shard` and an N=8 point on this tray
was not taken. So the numbers below are a LIKE-FOR-LIKE comparison -- the same two widths, the same
mesh, the same process, for the plain row shard and for the same shard with `b` split too -- and
the asymptote of a two-point fit is cruder than the parent's three-point one. What transfers is the
RATIO of the two asymptotes; the parent's better-supported 2.299x is rescaled by it and labelled.

    python3 perf/b2z2_shardrep/fit_bshard.py
"""

import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
PARENT_CAP_MEASURED_LINK = 2.299   # b2z2-trunk-shard-scale-wh, 3 points at 2/4/8
PARENT_CAP_FREE_LINK = 2.990
CELL_S = 20.113                    # site/data/perf-512aa.json @ 84da2a49b
TRUNK_S = 10.22                    # PairformerLayer device spans in that fold, CONTEXT 1-CORRECTION-D

runs = {}
for n in (2, 4):
    p = HERE / f"scale_bshard_512_n{n}.json"
    d = json.loads(p.read_text())["by_size"]["512"]
    runs[n] = {k: d[k]["median_ms"] for k in ("whole", "shard", "bshard")}

whole = sum(r["whole"] for r in runs.values()) / len(runs)


def fit(arm):
    """y = a + b/N through the two widths. `a` is the block wall no chip count removes."""
    (n1, y1), (n2, y2) = [(n, runs[n][arm]) for n in sorted(runs)]
    b = (y1 - y2) / (1 / n1 - 1 / n2)
    return y2 - b / n2, b


print(f"whole block (mean of the two meshes): {whole:.3f} ms")
print()
print(" N   whole     shard    bshard   shard/whole  bshard/whole  bshard/shard")
for n in sorted(runs):
    r = runs[n]
    print(f"{n:2d}  {r['whole']:7.3f}  {r['shard']:7.3f}  {r['bshard']:7.3f}   "
          f"{r['whole']/r['shard']:9.4f}x  {r['whole']/r['bshard']:10.4f}x  "
          f"{r['shard']/r['bshard']:10.4f}x")
print()
out = {"runs": runs, "whole_mean_ms": whole}
for arm in ("shard", "bshard"):
    a, b = fit(arm)
    cap = whole / a
    out[arm] = {"a_ms": a, "b_ms": b, "cap": cap, "replicated_pct": 100 * a / whole}
    print(f"{arm:7s} block(N) = {a:7.3f} ms + {b:7.3f} ms/N   -> cap {cap:.4f}x, "
          f"un-shardable {100*a/whole:5.1f} % of the block")
ratio = out["shard"]["a_ms"] / out["bshard"]["a_ms"]
out["cap_ratio"] = ratio
print()
print(f"the asymptote shrinks by {ratio:.4f}x, so the cap grows by the same factor")
print(f"rescaling the parent's THREE-point caps by it: "
      f"{PARENT_CAP_MEASURED_LINK * ratio:.3f}x measured link, "
      f"{PARENT_CAP_FREE_LINK * ratio:.3f}x free link (was "
      f"{PARENT_CAP_MEASURED_LINK}x / {PARENT_CAP_FREE_LINK}x)")
out["rescaled_cap_measured_link"] = PARENT_CAP_MEASURED_LINK * ratio
out["rescaled_cap_free_link"] = PARENT_CAP_FREE_LINK * ratio
print()
print("projected onto the published cell -- WH block ratios on BH seconds, an upper bound:")
for label, cap in (("shard, any N, measured link", PARENT_CAP_MEASURED_LINK),
                   ("bshard, any N, measured link", out["rescaled_cap_measured_link"]),
                   ("shard, any N, free link", PARENT_CAP_FREE_LINK),
                   ("bshard, any N, free link", out["rescaled_cap_free_link"])):
    trunk = TRUNK_S / cap
    fold = CELL_S - TRUNK_S + trunk
    print(f"  {label:32s} block {cap:6.3f}x  trunk {trunk:6.3f} s  fold {fold:7.3f} s  "
          f"{CELL_S / fold:.4f}x")
    out.setdefault("fold", {})[label] = {"block": cap, "trunk_s": trunk, "fold_s": fold,
                                         "fold_ratio": CELL_S / fold}
(HERE / "fit_bshard.json").write_text(json.dumps(out, indent=1))
print()
print(f"wrote {HERE / 'fit_bshard.json'}")
