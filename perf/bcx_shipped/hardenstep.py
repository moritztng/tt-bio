#!/usr/bin/env python3
"""Test converge.py's own premise: does anneal SPREAD predict the harden step?

converge.py was written asserting that anneal's spread over its last rounds "is what
separates a trajectory that survives harden from one that does not". That was a reading of
BindCraft 2's method, not a measurement, and this campaign has eight trajectories on disk
with the numbers to check it. If it does not hold, converge.py's docstring is a claim the
data refuses and it has to go, because the next reader will trust it.

One row per trajectory: the matched-k anneal spread, the matched-k anneal->harden step, the
stage it terminated at (blank = completed) and whether it was accepted. Association is
Spearman rho with an EXACT permutation p over all n! orderings -- n is 8 and 8! is 40320, so
there is no reason to reach for a normal approximation at a sample size where it is wrong.

usage: hardenstep.py <class>=<project> [<class>=<project> ...]
"""
import csv
import itertools
import math
import pathlib
import statistics
import sys

from converge import series, tail

STAGES = ("screen", "refine", "anneal", "harden", "mutate")


def terminated(proj):
    """-> {design: (terminated_stage, accepted)} from BindCraft 2's own summary CSV."""
    out = {}
    path = proj / "1_Trajectories" / "!_Trajectories.csv"
    if not path.is_file():
        return out
    ranked = set()
    rank_csv = proj / "3_Ranked" / "!_Ranked.csv"
    if rank_csv.is_file():
        with open(rank_csv) as fh:
            ranked = [r.get("design", "") for r in csv.DictReader(fh)]
    with open(path) as fh:
        for row in csv.DictReader(fh):
            design = row.get("design", "")
            # BindCraft 2 ranks a design as "<trajectory>_seq<n>" (campaign_output.py), one row
            # per accepted MPNN sequence, so the trajectory name is a PREFIX and never equal.
            out[design] = (row.get("terminated") or "",
                           sum(1 for d in ranked if d.startswith(design)))
    return out


def rows(label, project):
    proj = pathlib.Path(project)
    term = terminated(proj)
    for path in sorted(proj.glob("1_Trajectories/*/*_losses.csv")):
        stages = series(path)
        present = [s for s in STAGES if s in stages]
        if "anneal" not in present or "harden" not in present:
            continue
        k = min(len(stages[s]) for s in present)
        anneal = tail([v[1] for v in stages["anneal"]], k)
        harden = tail([v[1] for v in stages["harden"]], k)
        stage, accepted = term.get(path.parent.name, ("", 0))
        yield {"class": label, "draw": path.parent.name.split("_")[2],
               "spread": max(anneal) - min(anneal), "step": max(harden) - max(anneal),
               "reached": present[-1], "terminated": stage or "-", "accepted": accepted}


def rho(xs, ys):
    def rank(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        out = [0.0] * len(vs)
        i = 0
        while i < len(order):                      # average ranks within a tie group
            j = i
            while j + 1 < len(order) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            for t in range(i, j + 1):
                out[order[t]] = (i + j) / 2 + 1
            i = j + 1
        return out
    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else float("nan")


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    if not specs:
        sys.exit(__doc__)
    table = [r for label, project in specs for r in rows(label, project)]
    if not table:
        sys.exit("no trajectory with both an anneal and a harden stage on disk")
    print("  %-10s %5s %13s %14s %8s %10s  accepted"
          % ("class", "draw", "anneal spread", "anneal->harden", "reached", "terminated"))
    for r in sorted(table, key=lambda r: (r["class"], r["draw"])):
        print("  %-10s %5s %13.2f %+14.2f %8s %10s  %s"
              % (r["class"], r["draw"], r["spread"], r["step"], r["reached"],
                 r["terminated"], r["accepted"] or ""))
    xs = [r["spread"] for r in table]
    ys = [r["step"] for r in table]
    obs = rho(xs, ys)
    perms = list(itertools.permutations(range(len(ys))))
    hits = sum(abs(rho(xs, [ys[i] for i in p])) >= abs(obs) - 1e-12 for p in perms)
    p = hits / len(perms)
    print("\n  n = %d   Spearman rho = %+.3f   exact two-sided p = %.4f over %d permutations"
          % (len(table), obs, p, len(perms)))
    print("  premise: anneal spread predicts the harden STEP -- %s"
          % ("SUPPORTED" if p < 0.05 else "NOT SUPPORTED at this n"))

    # The other half of the premise, and the one that survives. "Do the accepted trajectories
    # hold the lowest anneal spreads" is a RANK question, so it is answerable without choosing a
    # threshold after seeing the data -- a cut at 0.07 would be tuned to this sample and would
    # not be evidence. Exact one-sided p = 1 / C(n, a): the chance that a of n trajectories,
    # placed at random, take the a lowest ranks.
    acc = sorted(r["spread"] for r in table if r["accepted"])
    rej = sorted(r["spread"] for r in table if not r["accepted"])
    if acc and rej:
        lowest = max(acc) < min(rej)
        pe = 1 / math.comb(len(table), len(acc))
        print("\n  accepted anneal spread %s   rejected %s"
              % ("/".join("%.2f" % v for v in acc), "/".join("%.2f" % v for v in rej)))
        print("  accepted hold the %d lowest spreads: %s   exact one-sided p = %.4f"
              % (len(acc), "YES" if lowest else "no", pe if lowest else float("nan")))


if __name__ == "__main__":
    main()
