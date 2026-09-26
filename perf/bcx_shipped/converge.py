#!/usr/bin/env python3
"""Read a stage at a MATCHED round count, and say whether anneal converged.

The run log prints each stage's BEST round. anneal runs 45 rounds and harden 5, so an
anneal max is a max over 45 draws and a harden max over 5: the printed anneal->harden step
is biased down whether or not anything changed (bcx-accept, STAGE-PROFILE.md).

This reads every stage over its LAST k rounds, k = the shortest stage's length, so both
stages and both arms are the same statistic. It also reports anneal's SPREAD over those
rounds. That spread was written up here as what separates a trajectory that survives harden
from one that does not. hardenstep.py then measured it on the nine trajectories this campaign
has on disk and it does NOT predict the harden step: Spearman rho -0.487, exact two-sided
p = 0.1869. What it does predict is ACCEPTANCE. The three accepted trajectories hold the three
lowest spreads across both arms, 0.03/0.06/0.07 against 0.08 to 0.67, exact one-sided
p = 0.0119. Read the spread as a read on the whole trajectory, not on harden.

usage: converge.py <project> [<project> ...]
"""
import csv
import pathlib
import statistics
import sys

STAGES = ("screen", "refine", "anneal", "harden", "mutate")
CONVERGED_BAR = 0.20


def series(path):
    """-> {stage: [(round, iptm, ptm), ...]} in file order."""
    out = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            phase = (row.get("phase") or "").strip()
            if not phase:
                continue
            ikey = next((k for k in row if k.endswith(".iptm")), None)
            pkey = next((k for k in row if k.endswith(".ptm")), None)
            try:
                iptm, ptm = float(row[ikey]), float(row[pkey])
            except (TypeError, ValueError, KeyError):
                continue
            out.setdefault(phase, []).append((row.get("round"), iptm, ptm))
    return out


def tail(values, k):
    return values[-k:] if k and len(values) >= k else values


def report(project):
    proj = pathlib.Path(project)
    csvs = sorted(proj.glob("1_Trajectories/*/*_losses.csv"))
    if not csvs:
        print(f"{proj}: no trajectory losses CSV")
        return
    for path in csvs:
        stages = series(path)
        present = [s for s in STAGES if s in stages]
        if not present:
            print(f"{path.parent.name}: no phase rows")
            continue
        k = min(len(stages[s]) for s in present)
        print(f"\n{path.parent.name}   matched on last k={k} rounds of each stage")
        print(f"  {'stage':8} {'n':>3}  {'med':>5} {'min':>5} {'max':>5} {'spread':>6}   full max")
        prev = None
        for name in present:
            window = tail([v[1] for v in stages[name]], k)
            med, lo, hi = statistics.median(window), min(window), max(window)
            step = "" if prev is None else f"   step {hi - prev:+.2f}"
            print(f"  {name:8} {len(stages[name]):>3}  {med:5.2f} {lo:5.2f} {hi:5.2f} "
                  f"{hi - lo:6.2f}   {max(v[1] for v in stages[name]):.2f}{step}")
            prev = hi
        if "anneal" in stages:
            window = tail([v[1] for v in stages["anneal"]], k)
            spread = max(window) - min(window)
            verdict = "no" if spread > CONVERGED_BAR else "yes"
            print(f"  anneal CONVERGED: {verdict} (spread {spread:.2f} over its last {k} "
                  f"rounds, bar {CONVERGED_BAR:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for project in sys.argv[1:]:
        report(project)
