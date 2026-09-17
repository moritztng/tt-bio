#!/usr/bin/env python3
"""Turn one measured session into the fold's per-class cycle budget, ranked by cycles above roof.

Reads a `replay.json` and nothing else from the device. Every number is derived from the
qualified brackets only: an interval whose during-samples were not min == max == 1350 MHz, or
that had a read error or a gap above 10 ms, contributes nothing.

Conversions, stated once:
  fold seconds for a key = its measured seconds per call x the call count the fold's own census
  recorded for that key. Mcycles = fold seconds x 1350. A rate is not a fold second until it
  has been through the call census.

Where a key was measured in more than one configuration the FASTER one is kept, so each rate
is an upper bound on what the shape can do and the fold seconds it implies are the smallest
this harness could make them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import census as C                                                            # noqa: E402

MHZ = 1350
BARE_FOLD_S = 14.881          # c10-bare-baseline pooled 512 aa median at a pinned 1350 MHz
INTERCEPT_S = 3.478           # the parent's re-fitted clock-immune term, fit_reexam
FRAMINGS = {"brief_15355_9584": (15355.0, 9584.0), "refit_14960_8805": (14960.0, 8805.0)}
MATMUL = ("linear", "matmul")
THREE_BYTE = ("multiply_", "layer_norm", "layer_norm_w", "add_")


ROOF_TOL = 1.0     # percent: the measured DRAM roof calibration residual, see refutation


def qualified(rows):
    """One row per key, the fastest qualified configuration of it."""
    best = {}
    for r in rows:
        if not r["clock_pass"] or r["s_per_call_qualified"] is None:
            continue
        if r["key"].startswith("roof."):
            continue
        cur = best.get(r["key"])
        if cur is None or r["s_per_call_qualified"] < cur["s_per_call_qualified"]:
            best[r["key"]] = r
    return best


def roofs(rows):
    """The compute and bandwidth roofs of THIS session, and the start/end cube agreement."""
    cubes = [r for r in rows if r["key"].startswith("roof.cube8192") and r["clock_pass"]]
    dram = [r for r in rows if r["key"].startswith("roof.dram_add") and r["clock_pass"]]
    rates = [r["flops_per_call"] / r["s_per_call_qualified"] / 1e12 for r in cubes]
    bw = [r["min_bytes_per_call"] / r["s_per_call_qualified"] / 1e9 for r in dram]
    drift = (100 * (max(rates) - min(rates)) / min(rates)) if len(rates) > 1 else 0.0
    other = [r for r in rows if r["key"] == "roof.cube4096" and r["clock_pass"]]
    return {"cube8192_TFLOPs": min(rates), "cube8192_readings": rates,
            "cube8192_drift_pct": drift,
            "cube4096_TFLOPs": (other[0]["flops_per_call"] / other[0]["s_per_call_qualified"] / 1e12
                                if other else None),
            "dram_GBs": max(bw) if bw else None,
            "known_answer": C.cube_control()}


def dispatch_floor(rows):
    """An upper bound on this session's own per-call host dispatch floor.

    The cheapest qualified arm in the session cannot be faster than one enqueue, so its per-call
    time bounds the floor from above. Any key whose measured per-call time is near it is
    dispatch-shaped, and its cycles above the arithmetic and traffic roofs are not evidence that
    a kernel is leaving work on the table. Launch is a THIRD roof and this row does not price
    it: perf/roof_launch owns that, at an unrecorded clock, which is why it is named here as a
    limitation rather than folded into a column.
    """
    vals = [r["s_per_call_qualified"] for r in rows
            if r["clock_pass"] and r["s_per_call_qualified"] and not r["key"].startswith("roof.")]
    return min(vals) if vals else None


def price(row, cube_TFLOPs, dram_GBs, floor_s=None):
    """Measured cost, and the cost the binding roof allows, for one key over the whole fold."""
    s = row["s_per_call_qualified"]
    fold_s = s * row["calls"]
    flops = row["flops_per_call"] * row["calls"]
    byts = row["min_bytes_per_call"] * row["calls"]
    arith_s = flops / (cube_TFLOPs * 1e12) if flops else 0.0
    traffic_s = byts / (dram_GBs * 1e9) if byts else 0.0
    roof_s = max(arith_s, traffic_s)
    return {"key": row["key"], "arm": row["arm"], "out": row["out"], "K": row["K"],
            "dispatch_shaped": bool(floor_s and s < 3 * floor_s),
            "us_per_call_over_dispatch_floor": s / floor_s if floor_s else None,
            "grid": row["grid"], "calls": row["calls"], "us_per_call": s * 1e6,
            "fold_s": fold_s, "Mcycles": fold_s * MHZ,
            "TFLOP": flops / 1e12, "GB": byts / 1e9,
            "TFLOPs": flops / fold_s / 1e12 if flops else None,
            "GBs": byts / fold_s / 1e9 if fold_s else None,
            "pct_of_cube": 100 * (flops / fold_s / 1e12) / cube_TFLOPs if flops else None,
            "pct_of_dram_roof": 100 * (byts / fold_s / 1e9) / dram_GBs if fold_s else None,
            "arith_roof_s": arith_s, "traffic_roof_s": traffic_s,
            "binding": ("arithmetic" if arith_s > traffic_s else
                        ("traffic" if traffic_s > 0 else "neither")),
            "roof_s": roof_s, "roof_Mcycles": roof_s * MHZ,
            "above_roof_s": fold_s - roof_s, "above_roof_Mcycles": (fold_s - roof_s) * MHZ}


def classes(priced):
    out = {}
    for p in priced:
        e = out.setdefault(p["arm"], {"arm": p["arm"], "keys": 0, "calls": 0.0, "fold_s": 0.0,
                                      "TFLOP": 0.0, "GB": 0.0, "roof_s": 0.0})
        e["keys"] += 1
        for k in ("calls", "fold_s", "TFLOP", "GB", "roof_s"):
            e[k] += p[k]
    for e in out.values():
        e["Mcycles"] = e["fold_s"] * MHZ
        e["above_roof_s"] = e["fold_s"] - e["roof_s"]
        e["above_roof_Mcycles"] = e["above_roof_s"] * MHZ
        e["TFLOPs"] = e["TFLOP"] / e["fold_s"] if e["fold_s"] else None
        e["GBs"] = e["GB"] / e["fold_s"] if e["fold_s"] else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    d = json.loads((a.run / "replay.json").read_text())
    rows = d["rows"]
    R = roofs(rows)
    if not (R["known_answer"]["flops_pass"] and R["known_answer"]["bytes_pass"]):
        raise RuntimeError("known-answer control failed; no table")
    best = qualified(rows)
    floor_s = dispatch_floor(rows)
    priced = sorted((price(r, R["cube8192_TFLOPs"], R["dram_GBs"], floor_s)
                     for r in best.values()), key=lambda p: -p["fold_s"])
    cls = classes(priced)

    shapes, op_census = C.load(ROOT / "perf")
    fold_calls = op_census["total_calls"]
    key_calls = sum(x["calls"] for x in shapes)
    covered = sum(p["calls"] for p in priced)
    mm = [p for p in priced if p["arm"] in MATMUL]
    mm_s = sum(p["fold_s"] for p in mm)
    mm_flop = sum(p["TFLOP"] for p in mm)
    three = [p for p in priced if p["arm"] in THREE_BYTE]
    total_s = sum(p["fold_s"] for p in priced)

    ladder = []
    for p in priced:
        if p["above_roof_Mcycles"] <= 0:
            continue
        row = {"key": p["key"], "arm": p["arm"], "calls": p["calls"], "binding": p["binding"],
               "measured_Mcycles": p["Mcycles"], "roof_Mcycles": p["roof_Mcycles"],
               "above_roof_Mcycles": p["above_roof_Mcycles"],
               "above_roof_s": p["above_roof_s"]}
        for name, (work, target) in FRAMINGS.items():
            row["pct_of_" + name] = 100 * p["above_roof_Mcycles"] / (work - target)
        ladder.append(row)
    ladder.sort(key=lambda r: -r["above_roof_Mcycles"])

    result = {
        "scope": ("Standalone replay of the fold's own recorded launch keys on qb2 node 1 at a "
                  "pinned, during-sampled 1350 MHz, weighted by the fold's own call census. "
                  "This is NOT an in-situ per-op measurement of the fold and no whole-fold "
                  "closure is claimed from it."),
        "node": d["preflight"]["node_identity_pre"], "device": d["device"],
        "clock": {"samples": d["clock_samples"], "min_MHz": d["clock_min_MHz"],
                  "max_MHz": d["clock_max_MHz"], "read_errors": d["clock_read_errors"],
                  "force_response": d["force_response"],
                  "release_response": d["release_response"],
                  "intervals_total": len(rows),
                  "intervals_qualified": sum(1 for r in rows if r["clock_pass"]),
                  "foreign_holder_observations": len(d["foreign_holders"]),
                  "holder_observations": d["holder_observations"]},
        "roofs": R,
        "byte_identity": {k: v for k, v in C.byte_identity(op_census).items() if k != "rows"},
        "coverage": {"fold_calls": fold_calls, "launch_key_calls": key_calls,
                     "measured_calls": covered,
                     "pct_of_fold_calls": 100 * covered / fold_calls,
                     "pct_of_launch_key_calls": 100 * covered / key_calls,
                     "keys_measured": len(priced), "keys_refused": d["preflight"]["refused"],
                     "arms_dropped": d["errors"]},
        "matmul_class": {
            "keys": len(mm), "calls": sum(p["calls"] for p in mm), "TFLOP": mm_flop,
            "fold_s": mm_s, "Mcycles": mm_s * MHZ,
            "achieved_TFLOPs": mm_flop / mm_s,
            "pct_of_cube": 100 * (mm_flop / mm_s) / R["cube8192_TFLOPs"],
            "modelled_TFLOPs_frontier": 19.88,
            "ratio_to_modelled": (mm_flop / mm_s) / 19.88,
            "pct_of_bare_fold": 100 * mm_s / BARE_FOLD_S},
        "three_byte_classes": {
            "arms": THREE_BYTE, "keys": len(three), "calls": sum(p["calls"] for p in three),
            "GB": sum(p["GB"] for p in three), "fold_s": sum(p["fold_s"] for p in three),
            "Mcycles": sum(p["fold_s"] for p in three) * MHZ,
            "achieved_GBs": (sum(p["GB"] for p in three) / sum(p["fold_s"] for p in three)
                             if three else None),
            "pct_of_dram_roof": (100 * (sum(p["GB"] for p in three)
                                        / sum(p["fold_s"] for p in three)) / R["dram_GBs"]
                                 if three else None),
            "in_house_roof_bracket_s": [2.074, 2.523]},
        "totals": {"measured_keys_fold_s": total_s, "measured_keys_Mcycles": total_s * MHZ,
                   "bare_fold_s": BARE_FOLD_S, "pct_of_bare_fold": 100 * total_s / BARE_FOLD_S,
                   "clock_immune_intercept_s": INTERCEPT_S,
                   "exceeds_bare_fold": total_s > BARE_FOLD_S},
        "dispatch": {
            "session_floor_us_per_call": floor_s * 1e6,
            "floor_key": min((r for r in rows if r["clock_pass"]
                              and r["s_per_call_qualified"] == floor_s), key=lambda r: r["key"],
                             default={}).get("key"),
            "dispatch_shaped_keys": sum(1 for p in priced if p["dispatch_shaped"]),
            "dispatch_shaped_fold_s": sum(p["fold_s"] for p in priced if p["dispatch_shaped"]),
            "note": ("launch is a third roof and is NOT priced here. A key within 3x of the "
                     "session's own dispatch floor is flagged; its cycles above the arithmetic "
                     "and traffic roofs are not evidence that a kernel is slow.")},
        "refutation": {
            "cube_in_allowed_range": 95.0 <= R["cube8192_TFLOPs"] <= 130.0,
            "cube_drift_within_2pct": R["cube8192_drift_pct"] <= 2.0,
            "no_arm_exceeds_the_cube": all(p["pct_of_cube"] is None or p["pct_of_cube"] <= 100
                                           for p in priced),
            "no_arm_exceeds_the_dram_roof": all(p["pct_of_dram_roof"] is None
                                                or p["pct_of_dram_roof"] <= 100 + ROOF_TOL
                                                for p in priced),
            "dram_roof_residual_pct": max((p["pct_of_dram_roof"] - 100 for p in priced
                                           if p["pct_of_dram_roof"]), default=None),
            "no_foreign_holder": len(d["foreign_holders"]) == 0,
            "every_interval_qualified": all(r["clock_pass"] for r in rows),
            "measured_keys_fit_inside_the_fold": total_s <= BARE_FOLD_S,
            "fired": [k for k, v in {
                "measured_keys_fit_inside_the_fold": total_s <= BARE_FOLD_S,
                "cube_in_allowed_range": 95.0 <= R["cube8192_TFLOPs"] <= 130.0,
                "cube_drift_within_2pct": R["cube8192_drift_pct"] <= 2.0,
                "no_foreign_holder": len(d["foreign_holders"]) == 0,
                "every_interval_qualified": all(r["clock_pass"] for r in rows),
                "no_arm_exceeds_the_cube": all(p["pct_of_cube"] is None
                                               or p["pct_of_cube"] <= 100 for p in priced),
                "no_arm_exceeds_the_dram_roof": all(p["pct_of_dram_roof"] is None
                                                    or p["pct_of_dram_roof"] <= 100 + ROOF_TOL
                                                    for p in priced),
            }.items() if not v],
            "consequence": ("The pre-registered criterion 'weighted fold seconds must fit inside "
                            "the 14.881 s bare fold' is what decides whether this replay is a "
                            "LOWER bound on fold cost. If it fired, at least one arm is running "
                            "a configuration the fold does not use, so no per-class rate from "
                            "this session may be quoted as what the fold's shapes can do.")},
        "classes": sorted(cls.values(), key=lambda e: -e["fold_s"]),
        "cycles_above_roof": ladder,
        "keys": priced,
    }
    out = a.out or (a.run / "budget.json")
    out.write_text(json.dumps(result, indent=1) + "\n")

    print("CUBE %.2f TFLOP/s (drift %.3f %%), DRAM %.1f GB/s, cube4096 %.2f"
          % (R["cube8192_TFLOPs"], R["cube8192_drift_pct"], R["dram_GBs"],
             R["cube4096_TFLOPs"]))
    print("coverage %.0f of %.0f fold calls (%.1f %%), %d keys"
          % (covered, fold_calls, 100 * covered / fold_calls, len(priced)))
    print()
    print("%-14s %5s %9s %9s %9s %8s %8s %9s" % ("class", "keys", "calls", "fold_s", "Mcycles",
                                                 "TFLOP/s", "GB/s", "aboveroof"))
    for e in result["classes"]:
        print("%-14s %5d %9.0f %9.4f %9.1f %8s %8s %9.1f"
              % (e["arm"], e["keys"], e["calls"], e["fold_s"], e["Mcycles"],
                 ("%.2f" % e["TFLOPs"]) if e["TFLOPs"] else "-",
                 ("%.1f" % e["GBs"]) if e["GBs"] else "-", e["above_roof_Mcycles"]))
    m = result["matmul_class"]
    print()
    print("MATMUL CLASS  %.3f TFLOP in %.4f s = %.2f TFLOP/s, %.1f %% of the session cube, "
          "%.2fx the modelled 19.88, %.1f %% of the 14.881 s fold"
          % (m["TFLOP"], m["fold_s"], m["achieved_TFLOPs"], m["pct_of_cube"],
             m["ratio_to_modelled"], m["pct_of_bare_fold"]))
    t = result["three_byte_classes"]
    print("BYTE CLASSES  %.3f TB in %.4f s = %.1f GB/s, %.1f %% of the measured DRAM roof"
          % (t["GB"] / 1000, t["fold_s"], t["achieved_GBs"], t["pct_of_dram_roof"]))
    print("TOTAL of measured keys: %.4f s, %.1f %% of the 14.881 s bare fold"
          % (total_s, result["totals"]["pct_of_bare_fold"]))
    print("dispatch floor %.2f us/call (%s); %d keys within 3x of it, %.4f fold s"
          % (result["dispatch"]["session_floor_us_per_call"], result["dispatch"]["floor_key"],
             result["dispatch"]["dispatch_shaped_keys"],
             result["dispatch"]["dispatch_shaped_fold_s"]))
    fired = result["refutation"]["fired"]
    print("REFUTATION fired: %s" % (", ".join(fired) if fired else "none"))
    print()
    print("CYCLES ABOVE ROOF, top 12 of %d" % len(ladder))
    for r in ladder[:12]:
        print("  %-42s %8.1f Mc above %-10s %5.1f %% / %5.1f %% of the two targets"
              % (r["key"], r["above_roof_Mcycles"], r["binding"],
                 r["pct_of_brief_15355_9584"], r["pct_of_refit_14960_8805"]))
    print("\nwrote %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
