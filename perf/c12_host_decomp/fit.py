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

# What the row actually has to close against, superseding F.
#
# `c12-profiled-fold` measured the device term in situ at a held, during-sampled 1350 MHz and
# weighted it by the fold's own integer call counts: 13.2090 s of 14.8810 s, leaving a non-device
# remainder of 1.6720 s, or 1.6489 s once ConfidenceHeadsDevice's <= 0.0231 s bound is taken off
# the host side. That remainder is the ceiling on ALL exposed host time in the fold, so the item
# sum is checked against it first. F = 3.9830 s stays in the table as a second, wider target: F
# exceeds the measured remainder by 2.31-2.33 s, and that excess is clock-immune DEVICE cost which
# no host line item can or should account for. Closing against F would mean padding the table by
# 2.3 s of work that is not host work.
# Source: perf/c12_profiled_fold/runs/composed.json on wk/c12-profiled-fold @ a63d6d8e3.
NONDEVICE_REMAINDER = {512: (1.6489, 1.6720), 298: (None, None)}
# What 12.5 s needs from host once every device lever lands. Pass 23's 1.0100 s was an
# arithmetic slip -- it re-priced the matmul core pin and added the difference to the residual,
# but that lever was never in the three-lever device book the residual is computed from -- and it
# is RETRACTED. The device book is silu 0.2843 + cond-hoist 0.2415 + reblock-delete 1.0062 =
# 1.5320 s, 12.5 s needs 2.3810 s, so the host owes 0.8490 s. It is a BAND because
# c12-reblock-delete's own prediction is a band, so the bar is reported against every arm.
REDUCIBLE_BAR_S = 0.8490
REDUCIBLE_BAR_BAND_S = {"reblock_optimistic": 0.6545, "reblock_central_6Z": 0.7562,
                        "reblock_central_5Z": 0.8490, "reblock_pessimistic": 0.9210}


# The regions whose exclusive seconds are host Python by construction: featurisation, the batch
# build, the CIF writer, and the exclusive bodies of the three nested callers that hold the fold
# together. Everything else in the tree wraps a device call, so its exclusive time is host glue
# around a device wait and is reported but not counted into the host sum.
HOST_ITEMS = ("prepare", "to_batch", "write_result", "predict_step", "forward", "sampler",
              "predict_step/forward", "predict_step/forward/sampler")


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


def closure(item_F, fold_F, F_reference, size=None, host_F=None):
    s = sum(item_F.values())
    out = {"item_F_sum_s": s, "fold_F_s": fold_F, "gap_vs_fold_s": s - fold_F,
           "gap_vs_fold_pct": 100.0 * (s - fold_F) / fold_F if fold_F else None,
           "F_reference_s": F_reference, "gap_vs_reference_s": s - F_reference,
           "gap_vs_reference_pct": 100.0 * (s - F_reference) / F_reference if F_reference else None}
    lo, hi = NONDEVICE_REMAINDER.get(int(size) if size else 0, (None, None))
    if lo is not None:
        target = host_F if host_F is not None else s
        out["nondevice_remainder_s"] = [lo, hi]
        out["host_item_sum_s"] = host_F
        out["gap_vs_remainder_s"] = [target - hi, target - lo]
        out["pct_of_remainder"] = [100.0 * target / hi, 100.0 * target / lo]
        out["over_remainder"] = target > hi
        out["reducible_bar_s"] = REDUCIBLE_BAR_S
        out["reducible_bar_band_s"] = REDUCIBLE_BAR_BAND_S
        out["bar_pct_of_remainder"] = {k: [100.0 * v / hi, 100.0 * v / lo]
                                       for k, v in REDUCIBLE_BAR_BAND_S.items()}
    return out


def cache_rebuild(rows, clocks=(int(F_HI), int(F_LO))):
    """Price the program-cache REBUILD: `bare` (clear ON, production) against `keepcache`.

    `tt_bio/boltz2.py:5385` clears and re-enables the device program cache on EVERY
    `Boltz2.forward` unless `TT_BIO_BOLTZ2_KEEP_PROGRAM_CACHE` is set. The clear call is
    microseconds (the `cacheclear` arm times it directly); what it can cost is the rebuild of
    every program the fold then uses, which a profiler books inside whichever device call touches
    each program first and which therefore appears in no per-op view and in no region tree. Only
    this A/B prices it, and only inside one warm process where both arms share a built cache.

    Reported against the session's own A/A floor -- the `bare` rep spread at the same clock. A
    delta smaller than that floor is not a result.
    """
    out = {}
    for c in clocks:
        b = [x["elapsed_s"] for x in rows if x["arm"] == "bare" and x["clock_MHz"] == c
             and x["valid"]]
        k = [x["elapsed_s"] for x in rows if x["arm"] == "keepcache" and x["clock_MHz"] == c
             and x["valid"]]
        wit = [x.get("keep_flag_queries") for x in rows if x["arm"] == "keepcache"
               and x["clock_MHz"] == c and x["valid"]]
        if not (b and k):
            out[str(c)] = {"error": "need both bare and keepcache at this clock"}
            continue
        # The floor is the WIDER of the two arms' own spreads, not `bare`'s alone. Taking only
        # the reference arm's spread called a -0.4154 s delta "resolved" at 800 MHz against a
        # 0.0465 s floor while the keepcache arm's own reps spanned 0.9486 s -- i.e. the effect
        # was smaller than the noise of the arm that was supposed to show it. A one-rep arm has
        # no spread at all, so it reports a floor of 0 and can resolve anything: that is flagged
        # rather than believed.
        aa = max(band(b), band(k))
        d = st.median(b) - st.median(k)
        thin = min(len(b), len(k)) < 2
        out[str(c)] = {
            "bare_s": st.median(b), "keepcache_s": st.median(k), "reps": [len(b), len(k)],
            "rebuild_s": d, "aa_floor_s": aa, "bare_band_s": band(b),
            "keepcache_band_s": band(k), "thin_reps": thin,
            "resolved": bool(abs(d) > aa and not thin),
            "witness_all_true": bool(wit) and all(q and all(q) for q in wit),
            "verdict": ("UNRESOLVED -- one arm has a single rep, so it has no spread to be "
                        "scored against" if thin else
                        "rebuild priced" if abs(d) > aa else
                        "UNRESOLVED -- smaller than this session's own A/A floor")}
    return out


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
            "closure": closure({k: v["F_s"] for k, v in leaves.items()}, fold["F_s"], F_ref,
                               size=size,
                               host_F=sum(v["F_s"] for k, v in leaves.items()
                                          if k in HOST_ITEMS or k.startswith("prepare/"))),
            "cache_rebuild": cache_rebuild(r["rows"]),
            "aa_floor_s": {str(c): band(bare[c]) for c in (int(F_HI), int(F_LO))},
            "cif_digests": sorted({d for x in r["rows"] if x["valid"]
                                   for d in (x.get("cif") or {}).values()}),
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
