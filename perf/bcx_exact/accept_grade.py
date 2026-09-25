#!/usr/bin/env python3
"""Grade the exact_training=OFF arm against bcx-exact's BAR, clause by clause.

The BAR was fixed in state/bcx-exact.md at 2026-09-25 18:5xZ, before any OFF number existed,
and it is reproduced here verbatim so the grading and the criterion sit in one file. Nothing
below chooses a threshold; every threshold is quoted from that text.

Two things this file is careful about, both of them ways a grader lies in the arm's favour:

1. THE UNIT IS A UNIQUE TRAJECTORY HASH. Several campaign logs are re-stamped copies of one
   run (ref_s1.log, ref_s1.stamped.log, refsnap/ref_run.log). Pooling them counts one
   trajectory three times and shrinks every interval it appears in.
2. THE REFERENCE SET IS ENUMERATED, NOT CHOSEN. `reference_logs()` globs every campaign
   artifact log on the box and keeps the ones that carry trajectories, minus the subject's
   own. The BAR named accept_s3 alone; widening after the numbers exist can only be honest
   if the widening is a complete sweep, so both readings are reported and the narrow one is
   never dropped.

Stage parsing is stage_profile.py's, at 85877f7de, including its INVALID draw table.
"""
import csv
import glob
import json
import os
import re
import statistics as st
import sys

STAGES = ("screen", "refine", "anneal", "harden", "mutate")
SATURATED = 0.995

RE_TRAJ = re.compile(r"=== trajectory (\d+) \| (\S+) \| accepted (\d+)/(\d+) ===")
RE_STAGE = re.compile(
    r"(passed|rejected at) (\w+) design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)"
    r"(?:\s+due to \[([^\]]*)\])?")
RE_FINAL = re.compile(
    r"trajectory rejected\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)(?:\s+due to \[([^\]]*)\])?")
RE_DRAW = re.compile(r"_l(\d+)_([0-9a-f]+)$")

# stage_profile.py's table, carried verbatim: draws DECISION-RULE.md rules out, keyed on the
# hash so a reading is excluded wherever it appears.
INVALID = {"3c946b4d257ac696": "stale mask, served l142's",
           "922fc15bad085ec1": "stale mask, served l142's",
           "3150367da865d53b": "i_pTM 1.0 for all 15 mutate rounds, pre-mask-fix tree"}

SUBJECT = "/home/ttuser/bcx_exact_art/traj_off_s3.log"
SUBJECT_PROJECT = "/home/ttuser/bcx_exact_art/traj_off_s3"
# The BAR's own reference, the five pre-instrument device trajectories of STAGE_REFERENCE.txt.
NARROW = ["/home/ttuser/bcx_accept_art/accept_s3.log",
          "/home/ttuser/bcx_accept_art/accept_s4.log"]


def reference_logs():
    """Every campaign log that carries trajectories, minus the subject's. A sweep, not a pick."""
    out = []
    for path in sorted(set(glob.glob("/home/ttuser/bcx_*art*/*.log")
                           + glob.glob("/home/ttuser/bcx_*art*/*/*.log"))):
        if os.path.realpath(path) == os.path.realpath(SUBJECT):
            continue
        try:
            text = open(path, errors="replace").read()
        except OSError:
            continue
        if RE_TRAJ.search(text):
            out.append(path)
    return out


def parse(path):
    """Trajectories in one log. Terminal is the stage it died at, or 'final'/'accepted'."""
    trajs, cur = [], None
    for line in open(path, errors="replace"):
        m = RE_TRAJ.search(line)
        if m:
            draw = RE_DRAW.search(m.group(2))
            cur = {"log": path, "name": m.group(2),
                   "length": int(draw.group(1)) if draw else None,
                   "hash": draw.group(2) if draw else m.group(2),
                   "stages": {}, "terminal": "in flight"}
            trajs.append(cur)
            continue
        if cur is None:
            continue
        m = RE_STAGE.search(line)
        if m:
            verdict, stage, iptm, plddt, _due = m.groups()
            if stage in STAGES:
                cur["stages"][stage] = (float(iptm), float(plddt))
                if verdict == "rejected at":
                    cur["terminal"] = stage
            continue
        if RE_FINAL.search(line):
            cur["terminal"] = "final"
        elif "ACCEPTED" in line and cur["terminal"] == "in flight":
            cur["terminal"] = "accepted"
    return trajs


def dedupe(trajs):
    """One row per draw hash. A re-stamped copy of a run is the same trajectory."""
    seen, out = {}, []
    for t in trajs:
        prev = seen.get(t["hash"])
        if prev is None:
            seen[t["hash"]] = t
            out.append(t)
        elif len(t["stages"]) > len(prev["stages"]):
            out[out.index(prev)] = t          # keep the copy that got furthest
            seen[t["hash"]] = t
    return out


def readings(trajs, stage, metric):
    """Stage readings, dropping INVALID draws and i_pTM saturated at the length ceiling."""
    idx = 0 if metric == "i_pTM" else 1
    vals = []
    for t in trajs:
        if t["hash"] in INVALID or stage not in t["stages"]:
            continue
        v = t["stages"][stage]
        if metric == "i_pTM" and v[0] >= SATURATED:
            continue
        vals.append(v[idx])
    return vals


def fisher_ge(a, b, c, d):
    """One-sided Fisher exact, P(X >= a) on the 2x2 [[a,b],[c,d]]. No scipy on this venv."""
    from math import comb
    n, k, m = a + b + c + d, a + c, a + b
    return sum(comb(k, i) * comb(n - k, m - i) for i in range(a, min(k, m) + 1)) / comb(n, m)


def main():
    subject = dedupe(parse(SUBJECT))
    refs_wide = dedupe([t for p in reference_logs() for t in parse(p)])
    refs_narrow = dedupe([t for p in NARROW for t in parse(p)])

    out = {"subject": {"log": SUBJECT, "trajectories": len(subject),
                       "rows": [{k: t[k] for k in ("name", "length", "hash", "terminal", "stages")}
                                for t in subject]},
           "reference_logs": reference_logs(),
           "reference_wide_trajectories": len(refs_wide),
           "reference_narrow_trajectories": len(refs_narrow),
           "clauses": {}}

    # ---- CLAUSE 1, the gate: "accepts at least one binder reading pLDDT >= 0.90,
    #      Backbone_Clashes == 0, Hotspot_Contact_Fraction == 1.0"
    ranked = glob.glob(os.path.join(SUBJECT_PROJECT, "3_Ranked", "*.csv"))
    binders = []
    for path in ranked:
        with open(path) as fh:
            for row in csv.DictReader(fh):
                binders.append(row)
    passing = [b for b in binders
               if float(b["pLDDT"]) >= 0.90 and float(b["Backbone_Clashes"]) == 0
               and float(b["Hotspot_Contact_Fraction"]) == 1.0]
    out["clauses"]["1_gate_binder"] = {
        "criterion": "pLDDT >= 0.90 and Backbone_Clashes == 0 and Hotspot_Contact_Fraction == 1.0",
        "accepted_binders": len(binders), "passing": len(passing),
        "best": ({k: passing[0][k] for k in
                  ("design", "length", "pLDDT", "i_pTM", "pTM", "i_pAE", "Backbone_Clashes",
                   "Hotspot_Contact_Fraction", "Interface_Residues", "Interface_BuriedArea",
                   "Binder_RMSD", "Unbound_Binder_pLDDT")} if passing else None),
        "verdict": "PASS" if passing else "FAIL"}

    # ---- CLAUSE 1, the sub-clause: "i_pTM >= 0.80 is required at every stage except mutate".
    #      Graded on the subject AND on every reference trajectory, because a sub-clause the
    #      unmodified loop fails too is measuring the sub-clause and not the arm.
    def stage_subclause(trajs):
        ok = bad = 0
        for t in trajs:
            graded = [(s, v) for s, v in t["stages"].items()
                      if s != "mutate" and t["hash"] not in INVALID and v[0] < SATURATED]
            if not graded:
                continue
            if all(v[0] >= 0.80 for _s, v in graded):
                ok += 1
            else:
                bad += 1
        return ok, bad
    s_ok, s_bad = stage_subclause(subject)
    r_ok, r_bad = stage_subclause(refs_wide)
    n_ok, n_bad = stage_subclause(refs_narrow)
    out["clauses"]["1_subclause_stage_iptm"] = {
        "criterion": "i_pTM >= 0.80 at every stage except mutate",
        "subject_pass": s_ok, "subject_fail": s_bad,
        "reference_wide_pass": r_ok, "reference_wide_fail": r_bad,
        "reference_narrow_pass": n_ok, "reference_narrow_fail": n_bad,
        "verdict": "FAIL" if s_bad else "PASS",
        "discriminating": (r_ok + n_ok) > 0 and (r_bad + n_bad) == 0}

    # ---- CLAUSE 2, as STAGE_REFERENCE.txt operationalised it before the run reported a stage:
    #      "the device's per-stage medians inside the reference's per-stage range, and no stage
    #      at which the device terminates and the reference never does".
    for label, ref in (("narrow", refs_narrow), ("wide", refs_wide)):
        cells, inside = [], True
        for stage in STAGES:
            for metric in ("i_pTM", "pLDDT"):
                sv, rv = readings(subject, stage, metric), readings(ref, stage, metric)
                if not sv or not rv:
                    continue
                med, lo, hi = st.median(sv), min(rv), max(rv)
                ok = lo <= med <= hi
                inside &= ok
                cells.append({"stage": stage, "metric": metric, "subject_median": round(med, 3),
                              "subject_n": len(sv), "reference_lo": lo, "reference_hi": hi,
                              "reference_median": round(st.median(rv), 3), "reference_n": len(rv),
                              "inside": ok,
                              "at_or_above_reference_median": med >= st.median(rv)})
        s_term = {t["terminal"] for t in subject} - {"in flight"}
        r_term = {t["terminal"] for t in ref} - {"in flight"}
        novel = sorted(s_term - r_term)
        out["clauses"][f"2_discriminator_{label}"] = {
            "criterion": "per-stage medians inside the reference range, and no terminal stage "
                         "the reference never reaches",
            "cells": cells, "all_medians_inside": inside,
            "medians_at_or_above_reference": sum(c["at_or_above_reference_median"] for c in cells),
            "medians_total": len(cells),
            "subject_terminals": sorted(s_term), "reference_terminals": sorted(r_term),
            "novel_terminal_stages": novel,
            "verdict": "PASS" if inside and not novel else "FAIL"}

    # The one criterion that fired against the narrow reference, priced. A trajectory dying at
    # screen is a normal BindCraft 2 outcome; the question is whether the OFF arm does it more.
    s_scr = sum(t["terminal"] == "screen" for t in subject)
    s_done = sum(t["terminal"] != "in flight" for t in subject)
    r_scr = sum(t["terminal"] == "screen" for t in refs_wide)
    r_done = sum(t["terminal"] != "in flight" for t in refs_wide)
    out["clauses"]["2_screen_termination_rate"] = {
        "subject": f"{s_scr}/{s_done}", "reference_wide": f"{r_scr}/{r_done}",
        "fisher_one_sided_p": round(fisher_ge(s_scr, s_done - s_scr,
                                              r_scr, r_done - r_scr), 4)}

    # ---- CLAUSE 3, the seed floor: "A NO-GO on clause 1 or 2 needs three seeds."
    out["clauses"]["3_seed_floor"] = {
        "criterion": "a NO-GO on clause 1 or 2 needs three seeds",
        "subject_seeds": 1, "no_go_available": False}

    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
