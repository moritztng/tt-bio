#!/usr/bin/env python3
"""Time each design stage of an arm, and every gradient step, from the arm's own artifacts.

BC2 prints a stage verdict but no clock. Two instruments can supply one:

``pool_selections.jsonl`` writes one timestamped record per gradient step. Checked rather than
assumed: ``profile_s1`` holds 188 records after its relaunch against 175 steps in its two
completed trajectories plus 13 into the third, and the arm log's ``[pool]`` count is 150 over the
same span because BC2 logs a *swap* only when the drawn model differs from the resident. So the
jsonl counts steps, the log counts swaps, and the jsonl is the exact timer.

``<arm>_progress.csv`` samples every 120 s. That bounds a boundary to one interval, which is 2 %
on a 50-step stage and 20 % on ``harden``'s 5 steps, so it is the fallback and the table says
when it was used.

One trap the jsonl carries: the out directory survives a relaunch, so ``pool_selections.jsonl``
and ``.campaign_state.json`` span qb2's 09:03Z reboot while the log does not (the launcher
truncates it). Records before the stamp's launch time belong to a trajectory that was killed
without a verdict and are dropped here -- counting them would time this run against another
run's clock.

Usage: stage_timeline.py <arm-name>      e.g. profile_s1, profile_s2_arm
"""
import calendar
import csv
import json
import os
import re
import sys
import time

ART = "/home/ttuser/bcx_mutate_art"

POOL = re.compile(r"^\[pool\]")
TRAJ = re.compile(r"^=== trajectory (\d+) \| (\S+)")
VERDICT = re.compile(
    r"^\s+(passed|rejected at) (\w+) design stage\s+i_pTM=([\d.]+)\s+pLDDT=([\d.]+)(.*)$"
)
# The stage order BC2 runs. `mutate` is forward-only and this row has never seen a post-fix
# trajectory reach it, so whether it emits a pool selection per round is UNVERIFIED: the step
# accounting below would drift after it and the table flags that rather than printing a number.
ORDER = ["screen", "refine", "anneal", "harden", "mutate"]


def epoch(utc):
    return calendar.timegm(time.strptime(utc, "%Y-%m-%dT%H:%M:%SZ"))


def hhmm(t):
    return time.strftime("%H:%M:%SZ", time.gmtime(t))


def stamp(name):
    out = {}
    path = os.path.join(ART, name + "_stamp.txt")
    if os.path.exists(path):
        for line in open(path):
            bits = line.split(None, 1)
            if len(bits) == 2:
                out[bits[0]] = bits[1].strip()
    return out


def stage_plan(name):
    """The per-stage step budget the arm itself recorded, not one assumed here."""
    head = ""
    with open(os.path.join(ART, name + ".log"), errors="replace") as fh:
        for line in fh:
            head += line
            if line.rstrip() == "}":
                break
    return json.loads(head)["stage_plan"]["per_stage"]


def steps(name, launched):
    """Timestamps of this run's gradient steps, one per pool selection."""
    path = os.path.join(ART, name, "pool_selections.jsonl")
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if launched is None or r["t"] >= launched:
            out.append(r["t"])
    return sorted(out)


def events(name):
    out = []
    with open(os.path.join(ART, name + ".log"), errors="replace") as fh:
        for line in fh:
            m = TRAJ.match(line)
            if m:
                out.append({"kind": "traj", "traj": int(m.group(1)), "draw": m.group(2)})
                continue
            m = VERDICT.match(line)
            if m:
                filt = re.search(r"due to \[([^\]]*)\]", m.group(5))
                out.append({"kind": "verdict", "stage": m.group(2),
                            "passed": m.group(1) == "passed",
                            "iptm": float(m.group(3)), "plddt": float(m.group(4)),
                            "filters": filt.group(1) if filt else ""})
    return out


def clocks(name):
    rows = []
    path = os.path.join(ART, name + "_progress.csv")
    if not os.path.exists(path):
        return rows
    for r in csv.DictReader(open(path)):
        try:
            rows.append((epoch(r["utc"]), r["aiclk"]))
        except (KeyError, ValueError):
            continue
    return rows


def clock_over(rows, t0, t1):
    """AICLK sampled DURING the window. A rate without its clock is not a measurement."""
    seen = {c for t, c in rows if t0 <= t <= t1 and c not in ("NA", "")}
    if not seen:
        return "n/a"
    lo, hi = min(seen, key=int), max(seen, key=int)
    return lo if lo == hi else f"{lo}-{hi}"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "profile_s1"
    st = stamp(name)
    launched = None
    if "launched" in st:
        launched = calendar.timegm(time.strptime(st["launched"], "%Y%m%dT%H%M%SZ"))
    plan = stage_plan(name)
    ts = steps(name, launched)
    rows = clocks(name)

    print(f"# {name}  tree {st.get('tree', 'wt/bcx-mutate')}  commit {st.get('commit','?')[:9]}  "
          f"card {st.get('card','?')}  seed {st.get('seed','?')}")
    print(f"# stage plan {plan}")
    print(f"# {len(ts)} gradient steps timed from pool_selections.jsonl since {st.get('launched','?')}")
    if ts:
        gaps = sorted(ts[i + 1] - ts[i] for i in range(len(ts) - 1))
        if gaps:
            print(f"# per-step gap s: median {gaps[len(gaps)//2]:.1f}  "
                  f"p10 {gaps[len(gaps)//10]:.1f}  p90 {gaps[9*len(gaps)//10]:.1f}  "
                  f"max {gaps[-1]:.1f}  (pooled over every draw size this run has taken)")
    print()
    print(f"{'traj':<5} {'stage':<8} {'steps':>5} {'verdict':<9} {'i_pTM':>6} {'pLDDT':>6} "
          f"{'filter':<8} {'start':>10} {'elapsed_s':>9} {'s/step':>7} {'aiclk':>9}")

    idx, traj, drift = 0, None, False
    for e in events(name):
        if e["kind"] == "traj":
            traj = e["traj"]
            print(f"--- trajectory {e['traj']} {e['draw']}")
            continue
        n = plan.get(e["stage"])
        if e["stage"] == "mutate":
            drift = True
        if n and idx + n <= len(ts) and not drift:
            t0, t1 = ts[idx], ts[idx + n - 1]
            dt = t1 - t0
            start, el = hhmm(t0), f"{dt:.0f}"
            per = f"{dt / max(n - 1, 1):.1f}"
            clk = clock_over(rows, t0, t1)
            idx += n
        else:
            start = el = per = clk = "n/a"
            if n:
                idx += n
        print(f"{traj:<5} {e['stage']:<8} {n if n else '?':>5} "
              f"{'passed' if e['passed'] else 'REJECTED':<9} {e['iptm']:>6} {e['plddt']:>6} "
              f"{e['filters']:<8} {start:>10} {el:>9} {per:>7} {clk:>9}")
    if drift:
        print("\n# a mutate stage was reached: its 15 rounds are forward-only and whether they")
        print("# emit a pool selection is unverified, so step accounting after it is suppressed.")


if __name__ == "__main__":
    main()
