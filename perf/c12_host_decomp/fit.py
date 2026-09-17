"""Two-clock decomposition: per line item, split its seconds into a clock-immune term and a
work term, and close the sum against F.

The whole point of the row. `T = F + C/f` holds for the fold as a whole to inside the A/A floor
(`c10-fixed-cost`: max cell-median residual 0.1035 s at 512 aa against a 0.1257 s floor). The
same form applied to ONE line item asks whether that item's seconds move when the clock moves:

    F_i = g_hi * t_i(f_hi) + g_lo * t_i(f_lo)      g_hi = f_hi/(f_hi-f_lo), g_lo = -f_lo/(f_hi-f_lo)
    C_i = (t_i(f_lo) - t_i(f_hi)) * f_hi*f_lo/(f_hi-f_lo)

At 1350/800 MHz that is F_i = 2.454545*t_hi - 1.454545*t_lo and C_i = 1963.636*(t_lo - t_hi)
Mcycles. Two properties make this the right instrument rather than a ratio eyeballed by hand:

  * it is LINEAR, so if the items partition the fold then sum(F_i) = F_fold identically. The
    closure check is then an arithmetic identity on the measured numbers, and any gap is a gap
    in the partition, not in the fit.
  * a pure host item has t_lo = t_hi, so F_i = t_i and C_i = 0. A pure device item has
    t_lo/t_hi = f_hi/f_lo, so F_i = 0 and C_i = t_hi*f_hi. The two ends are distinguishable
    without a threshold pulled out of the air.

Classification is scored against the item's OWN repeatability, never a fixed percentage: an item
whose two-clock difference is inside its own rep spread cannot be called either way, and is
reported as UNRESOLVED rather than rounded into the answer that suits the table.
"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path

F_HI, F_LO = 1350.0, 800.0


def gains(f_hi=F_HI, f_lo=F_LO):
    return f_hi / (f_hi - f_lo), -f_lo / (f_hi - f_lo)


def fixed_and_work(t_hi, t_lo, f_hi=F_HI, f_lo=F_LO):
    """Return (F seconds, C Mcycles) for one item from its two clock readings."""
    g_hi, g_lo = gains(f_hi, f_lo)
    F = g_hi * t_hi + g_lo * t_lo
    C = (t_lo - t_hi) * f_hi * f_lo / (f_hi - f_lo)
    return F, C


def band(values):
    """Repeatability of one item at one clock: the larger of the rep spread and stdev."""
    if len(values) < 2:
        return 0.0
    return max(max(values) - min(values), st.stdev(values))


def classify(t_hi_reps, t_lo_reps, f_hi=F_HI, f_lo=F_LO):
    """CLOCK-IMMUNE / CLOCK-SCALED / MIXED / UNRESOLVED for one item, against its own spread."""
    t_hi, t_lo = st.median(t_hi_reps), st.median(t_lo_reps)
    b = max(band(t_hi_reps), band(t_lo_reps))
    F, C = fixed_and_work(t_hi, t_lo, f_hi, f_lo)
    moved = t_lo - t_hi
    fully = t_hi * (f_hi / f_lo - 1.0)           # what a pure device item would have moved
    if abs(moved) <= b:
        verdict = "CLOCK-IMMUNE"                  # did not move outside its own repeatability
    elif abs(F) <= b:
        verdict = "CLOCK-SCALED"                  # moved by exactly what 1/f predicts
    elif moved > b and F > b:
        verdict = "MIXED"
    else:
        verdict = "UNRESOLVED"
    if b > 0 and abs(moved) <= b and abs(fully) <= b:
        verdict = "UNRESOLVED"                    # too small for either arm to see
    # An item's fixed term is the part of its own seconds that survives an infinite clock, so
    # 0 <= F_i <= t_i(fast) bounds it on physics alone. Both bounds are reachable -- pure host
    # sits at the top, pure device at 0 -- and a violation is not a small item, it is a wrong
    # one: the arms relabelled, the slow arm measured faster than the fast one, or two different
    # workloads compared. Say IMPOSSIBLE rather than print a number that cannot be a duration.
    physical = (-b - 1e-12) <= F <= (t_hi + b + 1e-12)
    if not physical:
        verdict = "IMPOSSIBLE"
    return {"t_hi_s": t_hi, "t_lo_s": t_lo, "reps_hi": len(t_hi_reps), "reps_lo": len(t_lo_reps),
            "band_s": b, "moved_s": moved, "pure_device_move_s": fully,
            "ratio": (t_lo / t_hi) if t_hi else None, "F_s": F, "C_Mcycles": C,
            "physical": physical, "verdict": verdict}


def tree_items(rows, key="excl_s"):
    """Per-item exclusive seconds from a list of region trees, one per rep."""
    items = {}
    for tree in rows:
        for path, v in tree.items():
            items.setdefault(path, []).append(v[key])
    return items


def closure(item_F, fold_F, F_reference):
    s = sum(item_F.values())
    return {"item_F_sum_s": s, "fold_F_s": fold_F, "gap_vs_fold_s": s - fold_F,
            "gap_vs_fold_pct": 100.0 * (s - fold_F) / fold_F if fold_F else None,
            "F_reference_s": F_reference, "gap_vs_reference_s": s - F_reference,
            "gap_vs_reference_pct": 100.0 * (s - F_reference) / F_reference if F_reference else None}


def scaling(F_512, F_298, expected=2.043):
    out = {}
    for k in sorted(set(F_512) | set(F_298)):
        a, b = F_512.get(k), F_298.get(k)
        out[k] = {"F_512_s": a, "F_298_s": b,
                  "ratio": (a / b) if (a and b and abs(b) > 1e-6) else None,
                  "expected": expected}
    return out


def _cli(paths):
    """Reduce one or more capture result.json files into the row's table."""
    out = {"sizes": {}, "gains": gains(), "form": "T = F + C/f"}
    for p in paths:
        r = json.loads(Path(p).read_text())
        size = str(r["size"])
        bare = {c: [x["elapsed_s"] for x in r["rows"] if x["arm"] == "bare"
                    and x["clock_MHz"] == c and x["valid"]] for c in (int(F_HI), int(F_LO))}
        if not (bare[int(F_HI)] and bare[int(F_LO)]):
            out["sizes"][size] = {"error": "both clock arms are needed and one is missing"}
            continue
        fold = classify(bare[int(F_HI)], bare[int(F_LO)])
        trees = {c: [x["tree"] for x in r["rows"] if x["arm"] == "regions"
                     and x["clock_MHz"] == c and x["valid"]] for c in (int(F_HI), int(F_LO))}
        hi, lo = tree_items(trees[int(F_HI)]), tree_items(trees[int(F_LO)])
        items = {}
        for k in sorted(set(hi) & set(lo)):
            items[k] = classify(hi[k], lo[k])
        leaves = {k: v for k, v in items.items()}
        unattr = {c: [x["unattributed_s"] for x in r["rows"] if x["arm"] == "regions"
                      and x["clock_MHz"] == c and x["valid"]] for c in (int(F_HI), int(F_LO))}
        if unattr[int(F_HI)] and unattr[int(F_LO)]:
            leaves["(unattributed)"] = classify(unattr[int(F_HI)], unattr[int(F_LO)])
        F_ref = r.get("F_reference_s")
        out["sizes"][size] = {
            "fold": fold, "items": leaves,
            "closure": closure({k: v["F_s"] for k, v in leaves.items()}, fold["F_s"], F_ref),
            "perturbation": {
                arm: {str(c): st.median([x["elapsed_s"] for x in r["rows"] if x["arm"] == arm
                                         and x["clock_MHz"] == c and x["valid"]] or [float("nan")])
                      for c in (int(F_HI), int(F_LO))}
                for arm in sorted({x["arm"] for x in r["rows"]})},
        }
    if "512" in out["sizes"] and "298" in out["sizes"] and "items" in out["sizes"]["512"]:
        out["scaling"] = scaling(
            {k: v["F_s"] for k, v in out["sizes"]["512"]["items"].items()},
            {k: v["F_s"] for k, v in out["sizes"]["298"]["items"].items()})
    return out


if __name__ == "__main__":
    print(json.dumps(_cli(sys.argv[1:]), indent=2))
