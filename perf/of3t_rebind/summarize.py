#!/usr/bin/env python3
"""One table over every arm, read from the artifacts rather than retyped."""
import json
import sys
from pathlib import Path

D = Path("perf/of3t_rebind")
ARMS = ["shipped", "shipped_aa2", "repin", "aa", "stale", "norebind", "permute", "zero"]

rows = []
for arm in ARMS:
    p = D / f"traj_{arm}.json"
    if not p.is_file():
        continue
    d = json.load(open(p))
    L = {r["k"]: r for r in d["our_step_log"]}
    per = {r["k"]: r for r in d["per_step"]}
    last = per[max(per)]
    rows.append({
        "arm": arm,
        "d1": per[1]["rel_d"],
        "k2": per[2]["rel_d"] if 2 in per else None,
        "k20": last["rel_d"],
        "d_ours_20": last["d_ours_norm"],
        "d_theirs_20": last["d_theirs_norm"],
        "worst": last.get("worst_tensor"),
        "worst_rel": last.get("worst_rel"),
        "exp": d["growth_k2_20"]["exponent"],
        "r2": d["growth_k2_20"]["r2"],
        "resolve_min": min(L[k]["tape_resolves_after_step"] for k in L),
        "resolve_max": max(L[k]["tape_resolves_after_step"] for k in L),
        "of_walked": L[1]["of_walked"],
        "gn1": L[1]["grad_norm"], "gn20": L[max(L)]["grad_norm"],
        "renorm": d["flag_reach"].get("_SOFTMAX_BW_RENORM"),
        "s": round(d["timing_s"]["total"], 1),
    })

print("%-12s %12s %12s %12s %9s %8s %7s %11s %11s %6s" %
      ("arm", "d_1", "k=2", "k=20", "resolve", "exp", "r2", "gn k=1", "gn k=20", "s"))
for r in rows:
    print("%-12s %12.6e %12.6e %12.6e %4d-%-4d %8.4f %7.3f %11.6e %11.6e %6.0f" %
          (r["arm"], r["d1"], r["k2"], r["k20"], r["resolve_min"], r["resolve_max"],
           r["exp"], r["r2"], r["gn1"], r["gn20"], r["s"]))
print()
for r in rows:
    print("%-12s worst@20 %-46s %s  ||d20_ours||=%.6e ||d20_theirs||=%.6e renorm=%s" %
          (r["arm"], r["worst"], ("%.3e" % r["worst_rel"]) if r["worst_rel"] is not None else "",
           r["d_ours_20"], r["d_theirs_20"], r["renorm"]))

# A/A: shipped against shipped_aa2 in a separate process, exactly.
a = json.load(open(D / "traj_shipped.json"))
b = json.load(open(D / "traj_shipped_aa2.json")) if (D / "traj_shipped_aa2.json").is_file() else None
if b is not None:
    diffs = [(x["k"], x["rel_d"], y["rel_d"], x["d_ours_norm"], y["d_ours_norm"])
             for x, y in zip(a["per_step"], b["per_step"])]
    bad = [d for d in diffs if d[1] != d[2] or d[3] != d[4]]
    print(f"\nA/A shipped vs shipped_aa2 (separate process): {len(diffs)-len(bad)} of "
          f"{len(diffs)} rungs bit-identical on rel_d AND d_ours_norm")
    for d in bad[:5]:
        print("   k=%d rel_d %.17g vs %.17g ; d_ours %.17g vs %.17g" % d)

r = json.load(open(D / "traj_repin.json")) if (D / "traj_repin.json").is_file() else None
if r is not None:
    diffs = [(x["k"], x["rel_d"], y["rel_d"]) for x, y in zip(a["per_step"], r["per_step"])]
    bad = [d for d in diffs if d[1] != d[2]]
    print(f"repin (harness rebind on top of the library fix) vs shipped: "
          f"{len(diffs)-len(bad)} of {len(diffs)} rungs bit-identical")
