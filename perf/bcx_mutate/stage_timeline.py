#!/usr/bin/env python3
"""Time each design stage of an arm by joining its log to its progress CSV.

BC2 prints a stage verdict but no clock, and the arm log's ``[pool]`` lines carry no time
either. The progress CSV carries both: a UTC stamp and the cumulative ``[pool]`` count at that
moment. So a verdict's position in the log, measured in cumulative pool swaps, interpolates to a
wall-clock time, and the difference between two verdicts is the stage's duration.

The join is exact in swaps and approximate in time: the CSV samples every 120 s, so a boundary
lands within one sampling interval. That is 2 % on ``screen`` (50 steps) and 20 % on ``harden``
(5 steps), and the table prints the bound rather than hiding it -- a 5-step stage timed against a
120 s sampler is not a measurement of its rate.

AICLK is reported per stage from the same CSV rows, sampled DURING the window, because a rate
without the clock it ran at is not comparable to any other rate.

Usage: stage_timeline.py <arm-name>      e.g. profile_s1, profile_s2_arm
"""
import csv
import json
import os
import re
import sys

ART = "/home/ttuser/bcx_mutate_art"

# BC2 draws a design model per gradient step from a pool of five, and the arm logs a swap only
# when the draw differs from the resident, so swaps run at about 4/5 of steps. The join does not
# assume that ratio -- it reads the per-stage step budget from the arm's own stamp -- but the
# ratio is why a stage's swap count is short of its step count.
POOL = re.compile(r"^\[pool\]")
TRAJ = re.compile(r"^=== trajectory (\d+) \| (\S+)")
VERDICT = re.compile(
    r"^\s+(passed|rejected at) (\w+) design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)(.*)$"
)


def stage_plan(name):
    """The per-stage step budget the arm itself recorded, not one assumed here."""
    with open(os.path.join(ART, name + ".log")) as fh:
        head = ""
        for line in fh:
            head += line
            if line.rstrip() == "}":
                break
    return json.loads(head)["stage_plan"]["per_stage"]


def events(name):
    """Every trajectory start and stage verdict, tagged with cumulative pool swaps."""
    out, swaps = [], 0
    with open(os.path.join(ART, name + ".log"), errors="replace") as fh:
        for line in fh:
            if POOL.match(line):
                swaps += 1
                continue
            m = TRAJ.match(line)
            if m:
                out.append({"kind": "traj", "traj": int(m.group(1)), "draw": m.group(2),
                            "swaps": swaps})
                continue
            m = VERDICT.match(line)
            if m:
                filt = re.search(r"due to \[([^\]]*)\]", m.group(5))
                out.append({"kind": "verdict", "stage": m.group(2),
                            "passed": m.group(1) == "passed",
                            "iptm": float(m.group(3)), "plddt": float(m.group(4)),
                            "filters": filt.group(1) if filt else "",
                            "swaps": swaps})
    return out


def curve(name):
    rows = []
    with open(os.path.join(ART, name + "_progress.csv")) as fh:
        for r in csv.DictReader(fh):
            try:
                rows.append((r["utc"], int(r["pool_swaps"]), r["aiclk"]))
            except (KeyError, ValueError):
                continue
    return rows


def to_epoch(utc):
    import calendar
    import time
    return calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))


def at(rows, swaps):
    """Interpolate a wall-clock time for a cumulative swap count.

    Returns (epoch, resolution_seconds, aiclk_at_that_row) or None when the count falls outside
    the sampled range -- which it does for anything logged before the watcher started, and
    saying None is the honest answer there rather than extrapolating.
    """
    prev = None
    for utc, s, clk in rows:
        if s >= swaps:
            if prev is None:
                return None
            put, ps, _ = prev
            span = s - ps
            t0, t1 = to_epoch(put), to_epoch(utc)
            frac = 0.0 if span == 0 else (swaps - ps) / span
            return t0 + frac * (t1 - t0), t1 - t0, clk
        prev = (utc, s, clk)
    return None


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "profile_s1"
    plan = stage_plan(name)
    rows = curve(name)
    evs = events(name)

    print(f"# {name}: per-stage timing, joined log position -> progress CSV")
    print(f"# stage plan {plan}")
    print(f"# CSV sampling {rows[1][1] and ''}every 120 s, so each boundary is +/- one interval")
    print()
    print(f"{'traj':<5} {'stage':<8} {'steps':>5} {'verdict':<9} {'i_pTM':>6} {'pLDDT':>6} "
          f"{'filter':<10} {'elapsed_s':>9} {'s/step':>7} {'+/-s':>5} {'aiclk':>6}")

    traj, prev_swaps = None, None
    for e in evs:
        if e["kind"] == "traj":
            traj, prev_swaps = e["traj"], e["swaps"]
            print(f"--- trajectory {e['traj']} {e['draw']}")
            continue
        steps = plan.get(e["stage"])
        a, b = at(rows, prev_swaps), at(rows, e["swaps"])
        if a and b and steps:
            dt = b[0] - a[0]
            res = max(a[1], b[1])
            el, per, pm = f"{dt:.0f}", f"{dt / steps:.1f}", f"{res:.0f}"
        else:
            el = per = pm = "n/a"
        clk = b[2] if b else "n/a"
        print(f"{traj:<5} {e['stage']:<8} {steps if steps else '?':>5} "
              f"{'passed' if e['passed'] else 'REJECTED':<9} {e['iptm']:>6} {e['plddt']:>6} "
              f"{e['filters']:<10} {el:>9} {per:>7} {pm:>5} {clk:>6}")
        prev_swaps = e["swaps"]


if __name__ == "__main__":
    main()
