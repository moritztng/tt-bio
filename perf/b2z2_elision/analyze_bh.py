#!/usr/bin/env python3
"""Read fold_ab.py's run log and answer the three questions the Blackhole row owes.

fold_ab.py's own summary is a single median ratio and a min/max A/A floor. That is enough for a
1.05x sampler effect and not enough for a 1.009x fold effect, so this reads the same `runs` list
and adds what the brief asks for: the WORST per-rep paired ratio, a position-split A/A floor for
both arms (the campaign's protocol, two base positions per rep), and a within-arm sha256 check
kept separate from the across-arm one, because Blackhole fold nondeterminism would show up as the
first failing while the second is meaningless.

Runs over a partial log: it only ever uses complete reps.
"""
from __future__ import annotations

import argparse, json, statistics as st
from pathlib import Path


def med(v):
    v = [x for x in v if x is not None]
    return st.median(v) if v else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    d = json.loads(a.runs.read_text())
    runs = d.get("runs", [])
    timed = [r for r in runs if not r.get("warmup")]

    # only complete reps: a rep is complete when it has both arms at both positions
    by_rep: dict[int, list] = {}
    for r in timed:
        by_rep.setdefault(r["rep"], []).append(r)
    order = [r["arm"] for r in by_rep.get(min(by_rep), [])] if by_rep else []
    full = {i: v for i, v in by_rep.items() if len(v) == len(order) and order}
    used = [r for i in sorted(full) for r in full[i]]

    off = [r for r in used if r["arm"] == "off"]
    on = [r for r in used if r["arm"] == "on"]
    o = {"reps_complete": len(full), "n_off": len(off), "n_on": len(on),
         "rep_order": order,
         "started_utc": d.get("env", {}).get("started_utc"),
         "card": d.get("env", {}).get("card"), "host": d.get("env", {}).get("host"),
         "steps": d.get("env", {}).get("steps"), "recycles": d.get("env", {}).get("recycles")}
    if not off or not on:
        print(json.dumps(o, indent=1)); return 1

    o["median_fold_s"] = {"off": round(med([r["fold_s"] for r in off]), 4),
                          "on": round(med([r["fold_s"] for r in on]), 4)}
    o["fold_ratio"] = round(o["median_fold_s"]["off"] / o["median_fold_s"]["on"], 5)
    o["fold_saved_s"] = round(o["median_fold_s"]["off"] - o["median_fold_s"]["on"], 4)

    sm = {k: round(med([r["sampler_ms_per_step"] for r in v]), 4)
          for k, v in (("off", off), ("on", on))}
    o["median_sampler_ms_per_step"] = sm
    o["sampler_ratio"] = round(sm["off"] / sm["on"], 5)
    ss = {k: round(med([r["stages_s"].get("sampler") for r in v]), 4)
          for k, v in (("off", off), ("on", on))}
    o["median_sampler_s"] = ss
    o["sampler_stage_ratio"] = round(ss["off"] / ss["on"], 5)

    # --- per-rep paired ratios: the lever has to win in most reps, not in one ---------
    per = []
    for i in sorted(full):
        v = full[i]
        fo, fn = med([r["fold_s"] for r in v if r["arm"] == "off"]), \
                 med([r["fold_s"] for r in v if r["arm"] == "on"])
        so, sn = med([r["sampler_ms_per_step"] for r in v if r["arm"] == "off"]), \
                 med([r["sampler_ms_per_step"] for r in v if r["arm"] == "on"])
        per.append({"rep": i, "fold_ratio": round(fo / fn, 5),
                    "sampler_ratio": round(so / sn, 5),
                    "off_s": round(fo, 3), "on_s": round(fn, 3),
                    "load1": max(r["loadavg1"] for r in v)})
    o["per_rep"] = per
    o["worst_rep_fold_ratio"] = min(p["fold_ratio"] for p in per)
    o["best_rep_fold_ratio"] = max(p["fold_ratio"] for p in per)
    o["reps_won"] = sum(1 for p in per if p["fold_ratio"] > 1.0)
    o["worst_rep_sampler_ratio"] = min(p["sampler_ratio"] for p in per)
    o["reps_won_sampler"] = sum(1 for p in per if p["sampler_ratio"] > 1.0)

    # --- A/A and B/B floors, split by position inside the rep -------------------------
    def floor(arm):
        v = [r for r in used if r["arm"] == arm]
        first = [r["fold_s"] for k, r in enumerate(v) if k % 2 == 0]
        second = [r["fold_s"] for k, r in enumerate(v) if k % 2 == 1]
        if not first or not second:
            return None
        return {"first_median_s": round(med(first), 4), "second_median_s": round(med(second), 4),
                "ratio": round(med(first) / med(second), 5),
                "spread_pct": round(100 * (max(r["fold_s"] for r in v) -
                                           min(r["fold_s"] for r in v)) /
                                    med([r["fold_s"] for r in v]), 2)}
    o["AA_floor_off"] = floor("off")
    o["BB_floor_on"] = floor("on")

    # --- parity, within-arm first ------------------------------------------------------
    sh = {k: sorted({r["cif_sha256"] for r in v}) for k, v in (("off", off), ("on", on))}
    o["cif_sha256"] = sh
    o["within_arm_identical"] = {k: len(v) == 1 for k, v in sh.items()}
    o["bit_exact_across_arms"] = len(sh["off"]) == 1 and sh["off"] == sh["on"]
    o["plddt"] = {k: sorted({round(float(r["plddt"]), 6) for r in v})
                  for k, v in (("off", off), ("on", on))}

    # --- the lever actually took the path it claims ------------------------------------
    o["gather_stats"] = {k: sorted({tuple(r["gather_stats"]) for r in v})
                         for k, v in (("off", off), ("on", on))}
    o["gather_stats"] = {k: [list(t) for t in v] for k, v in o["gather_stats"].items()}
    o["loadavg1_range"] = [min(r["loadavg1"] for r in used), max(r["loadavg1"] for r in used)]

    # --- carry to the published cell ---------------------------------------------------
    CELL = 20.079
    o["published_cell_s"] = CELL
    o["cell_carried_s"] = round(CELL / o["fold_ratio"], 4)
    o["cell_carried_worst_s"] = round(CELL / o["worst_rep_fold_ratio"], 4)

    print(json.dumps(o, indent=1))
    if a.out:
        a.out.write_text(json.dumps(o, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
