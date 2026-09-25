#!/usr/bin/env python3
"""The stage-by-stage trace of a BindCraft 2 campaign, with its configuration attached.

Reads the timestamped copy of the campaign's own stdout (`run.stamped.log`, one UTC stamp
per line) plus the CSV tables the campaign writes itself, and prints a per-trajectory
table: binder length, every stage line with the seconds it took, the stage the trajectory
was rejected at and the filters that fired, and the accepted count.

A count without the model pool that produced it is the conflation this row exists to end,
so ACCEPTED is printed next to the resolved `design_models` and `validation_models` off
`arm_stamp.json` and the campaign's own settings, never alone.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

STAMP = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (.*)$")
TRAJ = re.compile(r"^=== trajectory (\d+) \| (\S+) \| accepted (\d+)/(\d+) ===")
LEN = re.compile(r"_l(\d+)_")
OUTCOME = re.compile(r"^  (passed|rejected at|trajectory rejected|binder hallucination successful|"
                     r"binder optimization|induced fit|\d+ of \d+ redesigns)(.*)$")
# The MPNN candidates are indented one space deeper than the stage lines and carry the only
# per-candidate `failed [...]` list the campaign prints. Matching them is not cosmetic: the
# stage lines say a trajectory reached the final filters, these say which ones fired.
CANDIDATE = re.compile(r"^   (\d+)/(\d+)  (ACCEPTED|rejected)\s+(.*)$")
FAILED = re.compile(r"failed \[(.*)\]")


def parse_log(path):
    """`[{n, name, binder_length, start, lines: [(t, dt_s, text)], end}]`."""
    trajectories, current = [], None
    prev_t = None
    for raw in open(path, errors="replace"):
        m = STAMP.match(raw.rstrip("\n"))
        if not m:
            continue
        t = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
        text = m.group(2)
        start = TRAJ.match(text)
        if start:
            current = {"n": int(start.group(1)), "name": start.group(2),
                       "accepted_before": int(start.group(3)),
                       "target": int(start.group(4)), "start": t, "lines": []}
            ln = LEN.search(start.group(2))
            current["binder_length"] = int(ln.group(1)) if ln else None
            trajectories.append(current)
            prev_t = t
            continue
        if current is None or not (OUTCOME.match(text) or CANDIDATE.match(text)):
            continue
        current["lines"].append((t, (t - prev_t).total_seconds(), text.strip()))
        prev_t = t
    return trajectories


def campaign_count(project, ranked):
    """The accepted count, off the campaign's own state file, cross-checked against 3_Ranked.

    BindCraft 2 never writes `accepted.csv`; it keeps the running count in
    `.campaign_state.json` and rewrites `3_Ranked/!_Ranked.csv` as each design is accepted.
    Reading a file the campaign does not write reports 0 accepted for every campaign, which
    is the one number this row exists to get right, so the two sources are compared and a
    disagreement is printed rather than silently resolved.
    """
    path = os.path.join(project, ".campaign_state.json")
    state = json.load(open(path)) if os.path.exists(path) else {}
    count = state.get("accepted")
    note = ""
    if count is None:
        count, note = len(ranked), "  (no .campaign_state.json; counted 3_Ranked rows)"
    elif count != len(ranked):
        note = f"  MISMATCH: .campaign_state.json {count} vs {len(ranked)} ranked rows"
    return count, note, state.get("rejections", {})


def rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def resolved_pool(stamp, project):
    """The pools BindCraft 2's own resolver gives, banked beside the run.

    A count is meaningless without them, so they are resolved with the same call
    `bindcraft/preflight.py:21` makes rather than read off the settings JSON, and cached in
    `model_pool.json` so the trace still carries them on a host without BindCraft 2.
    """
    cache = os.path.join(project, "model_pool.json")
    if os.path.exists(cache):
        pools = json.load(open(cache))
    else:
        sys.path.insert(0, stamp["bc2"])
        from bindcraft.af2 import MONOMER_POOL, MULTIMER_POOL
        from bindcraft.preflight import cleaned_campaign_settings
        from bindcraft.settings import (parse_setting_overrides, read_settings,
                                        select_design_and_validation_models)
        overrides = [] if stamp.get("shipped_model_pool") else [
            "validation_model=monomer", 'design_models=["model_1_ptm"]',
            'validation_models=["model_2_ptm"]']
        settings = cleaned_campaign_settings(
            read_settings(stamp["settings_file"], parse_setting_overrides(overrides)))
        selected = select_design_and_validation_models(settings, MULTIMER_POOL, MONOMER_POOL)
        pools = {"design_models": list(selected.design_models),
                 "validation_models": list(selected.validation_models)}
        json.dump(pools, open(cache, "w"), indent=1)
    stamp.setdefault("resolved_validation_models", pools["validation_models"])
    return pools["design_models"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    P = args.project
    stamp = json.load(open(os.path.join(P, "arm_stamp.json")))
    log = os.path.join(P, "run.stamped.log")
    trajectories = parse_log(log) if os.path.exists(log) else []

    ranked = rows(os.path.join(P, "3_Ranked", "!_Ranked.csv"))
    refolded = rows(os.path.join(P, "2_Refolded", "!_Refolded.csv"))
    traj_csv = rows(os.path.join(P, "1_Trajectories", "!_Trajectories.csv"))

    pool = stamp.get("resolved_design_models") or resolved_pool(stamp, P)
    accepted, count_note, rejections = campaign_count(P, ranked)
    out = {
        "project": P, "bc2": stamp.get("bc2"), "settings_file": stamp.get("settings_file"),
        "shipped_model_pool": stamp.get("shipped_model_pool"),
        "length_bucket_size": stamp.get("length_bucket_size"),
        "host": stamp.get("host"), "started_utc": stamp.get("started_utc"),
        "stage_plan": stamp.get("stage_plan"),
        "design_models": pool, "validation_models": stamp.get("resolved_validation_models"),
        "ACCEPTED": accepted, "ranked_rows": len(ranked), "rejections": rejections,
        "refolded_rows": len(refolded), "trajectory_rows": len(traj_csv),
        "trajectories": [],
    }
    for t in trajectories:
        last = t["lines"][-1] if t["lines"] else None
        out["trajectories"].append({
            "n": t["n"], "name": t["name"], "binder_length": t["binder_length"],
            "start_utc": t["start"].strftime("%FT%TZ"),
            "elapsed_s": round((last[0] - t["start"]).total_seconds(), 1) if last else None,
            "stages": [{"utc": a.strftime("%FT%TZ"), "seconds": round(b, 1), "line": c}
                       for a, b, c in t["lines"]],
            "outcome": last[2] if last else "in progress",
        })

    if args.json:
        json.dump(out, sys.stdout, indent=1)
        print()
        return
    print(f"bc2                {out['bc2']}")
    print(f"settings           {out['settings_file']}")
    print(f"shipped pool       {out['shipped_model_pool']}   bucket {out['length_bucket_size']}")
    print(f"design_models      {out['design_models']}")
    print(f"validation_models  {out['validation_models']}")
    print(f"ACCEPTED           {out['ACCEPTED']}   "
          f"(trajectory rows {out['trajectory_rows']}, refolded {out['refolded_rows']}, "
          f"ranked {out['ranked_rows']}){count_note}")
    if rejections:
        print(f"candidates         scored {rejections.get('candidates_scored')}, "
              f"rejected {rejections.get('candidates_rejected')}, "
              f"failed filters {rejections.get('failed_filters')}")
    for t in out["trajectories"]:
        print(f"\ntrajectory {t['n']}  binder {t['binder_length']}aa  start {t['start_utc']}  "
              f"elapsed {t['elapsed_s']}s")
        for s in t["stages"]:
            print(f"  +{s['seconds']:>8.1f}s  {s['line']}")


if __name__ == "__main__":
    main()
