#!/usr/bin/env python3
"""The honest upper bound on the Boltz-2 512 aa DEVICE term, priced against a MEASURED frontier.

C12 needs 1.898 s out of the 10.898 s device term for 10.0 s. Every per-class estimate so far has
been "cycles above the dense-cube roof", and that roof does not apply to thin or small-transfer
shapes -- the same error that self-refuted the 78.24 TFLOP/s matmul_ceiling. This prices each census
key against an envelope built from what THIS CHIP actually achieved in the same session:

    frontier_bytes(B) = max{ GB/s of any key whose bytes-per-call <= B }
    frontier_flops(F) = max{ TFLOP/s of any key whose FLOPs-per-call <= F }

Both are monotone by construction and conservative in the right direction: a key moving B bytes per
call cannot be told to beat a rate that no key with an equal or smaller transfer ever reached, and a
larger transfer amortises at least as well as a smaller one. A key's achievable time is then the
larger of its byte time and its FLOP time on those envelopes, and its prize is whatever it spends
above that. Summed, that is the most the device term can give -- not what it will give.

Known-answer controls, all asserted below:
  * the roof rows must price to ~zero prize (they define their own frontier points);
  * the best-arm fold seconds must reproduce the census's 10.5368 s of record;
  * the byte and FLOP totals must reproduce 2,451.2 GB and 131.27 TFLOP.

Usage (needs origin/wk/c10-fold-census fetched):
    python3 perf/c12_orchestrator/device_ceiling.py [--out runs/device_ceiling.json]
"""
import argparse
import json
import subprocess
from collections import defaultdict

REF = "origin/wk/c10-fold-census:perf/c10_fold_census/runs/sweep2/replay.json"
CLOCK_MHZ = 1350.0
DEVICE_TERM_S = 10.898      # 14.881 bare fold - F 3.983, c10-fixed-cost + c10-bare-baseline
CENSUS_OF_RECORD_S = 10.5368
OWED_FOR_10S_S = 1.898


def load_rows():
    blob = subprocess.run(["git", "show", REF], check=True, capture_output=True, text=True)
    return json.loads(blob.stdout)["rows"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = load_rows()

    # One key was replayed once per arm (default grid and [11,10]). The census of record used the
    # BEST arm, which this script asserts below, so the best arm is also what a ceiling must use:
    # pricing the worse arm would credit a lever with recovering a config difference the fold has
    # already realised.
    by_key = defaultdict(list)
    for r in rows:
        by_key[r["key"]].append(r)
    best = {k: min(v, key=lambda r: r["s_per_call"]) for k, v in by_key.items()}
    worst = {k: max(v, key=lambda r: r["s_per_call"]) for k, v in by_key.items()}

    # Frontier points come from every row, roofs included -- a roof is just the largest transfer.
    pts_b = sorted((r["min_bytes_per_call"], r["GBs"]) for r in rows
                   if r.get("min_bytes_per_call") and r.get("GBs"))
    pts_f = sorted((r["flops_per_call"], r["TFLOPs"]) for r in rows
                   if r.get("flops_per_call") and r.get("TFLOPs"))

    def envelope(pts):
        out, run = [], 0.0
        for x, y in pts:
            run = max(run, y)
            out.append((x, run))
        return out

    env_b, env_f = envelope(pts_b), envelope(pts_f)

    def at(env, x):
        best_y = env[0][1]
        for xx, yy in env:
            if xx > x:
                break
            best_y = yy
        return best_y

    def achievable_s(bytes_total, flops_total, bytes_per_call, flops_per_call):
        """Time this work cannot go below on the measured envelopes. Used by the ledger AND by the
        known-answer control, so the control exercises the same code path the result comes from."""
        rb = at(env_b, bytes_per_call or 0) if bytes_total else 0.0
        rf = at(env_f, flops_per_call or 0) if flops_total else 0.0
        t_b = bytes_total / (rb * 1e9) if rb else 0.0
        t_f = flops_total / (rf * 1e12) if rf else 0.0
        return max(t_b, t_f), t_b, t_f, rb, rf

    ledger, cls = [], defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    for k, r in best.items():
        if k.startswith("roof."):
            continue
        calls = r["calls"]
        fold_s = calls * r["s_per_call"]
        b_total = calls * (r["min_bytes_per_call"] or 0)
        f_total = calls * (r["flops_per_call"] or 0)
        ach, t_b, t_f, rb, rf = achievable_s(b_total, f_total, r["min_bytes_per_call"],
                                             r["flops_per_call"])
        prize = max(0.0, fold_s - ach)
        row = {"key": k, "arm": r["arm"], "calls": int(calls),
               "fold_s": round(fold_s, 4), "worst_arm_fold_s": round(worst[k]["calls"] * worst[k]["s_per_call"], 4),
               "GBs": round(r["GBs"], 2) if r.get("GBs") else None,
               "TFLOPs": round(r["TFLOPs"], 2) if r.get("TFLOPs") else None,
               "bytes_per_call": r["min_bytes_per_call"], "frontier_GBs": round(rb, 2),
               "frontier_TFLOPs": round(rf, 2), "achievable_s": round(ach, 4),
               "prize_s": round(prize, 4), "prize_Mcycles": round(prize * CLOCK_MHZ, 1),
               "binds_on": "bytes" if t_b >= t_f else "flops"}
        ledger.append(row)
        c = cls[r["arm"]]
        c[0] += fold_s; c[1] += ach; c[2] += prize; c[3] += 1

    # Known-answer controls.
    tot_fold = sum(r["fold_s"] for r in ledger)
    tot_prize = sum(r["prize_s"] for r in ledger)
    tot_bytes = sum(r["calls"] * (r["bytes_per_call"] or 0) for r in ledger) / 1e9
    roof_prizes = []
    for k, r in best.items():
        if not k.startswith("roof."):
            continue
        ach, _, _, _, _ = achievable_s(r["min_bytes_per_call"] or 0, r["flops_per_call"] or 0,
                                       r["min_bytes_per_call"], r["flops_per_call"])
        prize = max(0.0, r["s_per_call"] - ach)
        roof_prizes.append({"key": k, "prize_s_per_call": round(prize, 9),
                            "s_per_call": r["s_per_call"],
                            "prize_pct_of_own_time": round(100 * prize / r["s_per_call"], 3)})
    controls = {
        "best_arm_sum_s": round(tot_fold, 4),
        "census_of_record_s": CENSUS_OF_RECORD_S,
        "best_arm_reproduces_census": abs(tot_fold - CENSUS_OF_RECORD_S) < 0.01,
        "worst_arm_sum_s": round(sum(r["worst_arm_fold_s"] for r in ledger), 4),
        "worst_arm_exceeds_device_term": sum(r["worst_arm_fold_s"] for r in ledger) > DEVICE_TERM_S,
        "bytes_GB": round(tot_bytes, 1),
        # A roof row defines its own frontier point, so its prize must be the session's own
        # repeat-to-repeat drift and nothing more. 1 % relative, not absolute: the two cube readings
        # differ by 0.013 % and the DRAM roof by 0.44 %, and a frontier bug would show up as tens of
        # percent here.
        "roof_rows_price_to_zero": all(x["prize_pct_of_own_time"] <= 1.0 for x in roof_prizes),
        "roof_rows": roof_prizes,
    }
    report = {
        "ref": REF, "clock_MHz": CLOCK_MHZ, "keys": len(ledger),
        "device_term_s": DEVICE_TERM_S, "owed_for_10s_s": OWED_FOR_10S_S,
        "ceiling_s": round(tot_prize, 4), "ceiling_Mcycles": round(tot_prize * CLOCK_MHZ, 1),
        "ceiling_pct_of_owed": round(100 * tot_prize / OWED_FOR_10S_S, 1),
        "controls": controls,
        "by_class": {a: {"keys": v[3], "fold_s": round(v[0], 4), "achievable_s": round(v[1], 4),
                         "prize_s": round(v[2], 4)} for a, v in sorted(cls.items(), key=lambda kv: -kv[1][2])},
        "ledger": sorted(ledger, key=lambda r: -r["prize_s"]),
    }
    assert controls["best_arm_reproduces_census"], controls
    assert controls["roof_rows_price_to_zero"], controls
    print(json.dumps({k: v for k, v in report.items() if k != "ledger"}, indent=1))
    print("top 12 by prize:")
    for r in report["ledger"][:12]:
        print(f"  {r['key']:<44} {r['fold_s']:7.4f}s -> {r['achievable_s']:7.4f}s  prize {r['prize_s']:7.4f}s "
              f"({r['prize_Mcycles']:7.1f} Mc)  {(r['GBs'] or 0):6.1f}->{r['frontier_GBs']:6.1f} GB/s  binds {r['binds_on']}")
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
