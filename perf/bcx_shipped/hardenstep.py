#!/usr/bin/env python3
"""Test converge.py's own premise: does anneal SPREAD predict the harden step?

converge.py was written asserting that anneal's spread over its last rounds "is what
separates a trajectory that survives harden from one that does not". That was a reading of
BindCraft 2's method, not a measurement, and this campaign has eight trajectories on disk
with the numbers to check it. If it does not hold, converge.py's docstring is a claim the
data refuses and it has to go, because the next reader will trust it.

One row per trajectory: the matched-k anneal spread and median, the matched-k anneal->harden
step, the stage it terminated at (blank = completed) and whether it was accepted. Association
is Spearman rho with an EXACT permutation p over all n! orderings -- n is 9 and 9! is 362880,
so there is no reason to reach for a normal approximation at a sample size where it is wrong.

Spread is reported next to anneal's MEDIAN on purpose. The two are not independent: a
trajectory still bimodal at round 45 has a low median because half its rounds sit in the low
mode, so a separation on spread and a separation on level are the same separation until
something distinguishes them. Every candidate that splits accepted from rejected is ranked
here rather than only the one that was looked for.

usage: hardenstep.py <class>=<project> [<class>=<project> ...]
"""
import csv
import itertools
import math
import pathlib
import random
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
               "spread": max(anneal) - min(anneal), "median": statistics.median(anneal),
               "step": max(harden) - max(anneal), "len": int(path.parent.name.split("_")[2][1:]),
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



def perm_p(stat, labels, exact_cap=1_000_000, draws=200_000, seed=20260926):
    """Two-sided permutation p for `stat(labels)`, exact when n! fits, else Monte Carlo.

    n! is 3,628,800 at n=10 and 5.1e19 at n=21, so an "exact over all orderings" test
    silently stops being computable exactly when this campaign's sample got useful.
    Monte Carlo keeps the same null -- relabelings are equally likely -- and its own
    error is reported rather than hidden: the standard error of a proportion, and the
    +1/+1 estimator (Davison & Hinkley) so a p is never reported as 0.
    """
    obs = stat(labels)
    n = len(labels)
    if math.factorial(n) <= exact_cap:
        perms = itertools.permutations(labels)
        hits = sum(abs(stat(list(q))) >= abs(obs) - 1e-12 for q in perms)
        total = math.factorial(n)
        return obs, hits / total, total, "exact", 0.0
    rng = random.Random(seed)
    shuf, hits = list(labels), 0
    for _ in range(draws):
        rng.shuffle(shuf)
        hits += abs(stat(shuf)) >= abs(obs) - 1e-12
    pv = (hits + 1) / (draws + 1)
    return obs, pv, draws, "monte-carlo", math.sqrt(max(pv, 1e-9) * (1 - pv) / draws)


def mann_whitney_u(a, b):
    """U for group `a` against `b`, ties averaged. Rank-based, so no threshold is chosen."""
    allv = list(a) + list(b)
    order = sorted(range(len(allv)), key=lambda i: allv[i])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    ra = sum(ranks[: len(a)])
    return ra - len(a) * (len(a) + 1) / 2


def main():
    specs = [a.split("=", 1) for a in sys.argv[1:]]
    if not specs:
        sys.exit(__doc__)
    table = [r for label, project in specs for r in rows(label, project)]
    if not table:
        sys.exit("no trajectory with both an anneal and a harden stage on disk")
    print("  %-10s %5s %13s %10s %14s %8s %10s  accepted"
          % ("class", "draw", "anneal spread", "anneal med", "anneal->harden", "reached",
             "terminated"))
    for r in sorted(table, key=lambda r: (r["class"], r["draw"])):
        print("  %-10s %5s %13.2f %10.2f %+14.2f %8s %10s  %s"
              % (r["class"], r["draw"], r["spread"], r["median"], r["step"], r["reached"],
                 r["terminated"], r["accepted"] or ""))
    xs = [r["spread"] for r in table]
    ys = [r["step"] for r in table]
    obs, p, total, how, se = perm_p(lambda perm: rho(xs, perm), ys)
    print("\n  n = %d   Spearman rho = %+.3f   two-sided p = %.4f  (%s, %d relabelings%s)"
          % (len(table), obs, p, how, total, "" if how == "exact" else ", se %.4f" % se))
    print("  premise: anneal spread predicts the harden STEP -- %s"
          % ("SUPPORTED" if p < 0.05 else "NOT SUPPORTED at this n"))

    # The other half of the premise, and the one that survives. "Do the accepted trajectories
    # hold the lowest anneal spreads" is a RANK question, so it is answerable without choosing a
    # threshold after seeing the data -- a cut at 0.07 would be tuned to this sample and would
    # not be evidence. Exact one-sided p = 1 / C(n, a): the chance that a of n trajectories,
    # placed at random, take the a lowest ranks.
    n_acc = sum(1 for r in table if r["accepted"])
    if n_acc and n_acc < len(table):
        n_rej = len(table) - n_acc
        print("\n  separation of the %d accepted from the %d rejected, per candidate variable."
              % (n_acc, n_rej))
        print("  Mann-Whitney U on RANKS, so no threshold is chosen after seeing the data, with a\n"
              "  two-sided permutation p under the same null the earlier version used: the accepted\n"
              "  labels are exchangeable. PERFECT is the stricter all-or-nothing question the n=9\n"
              "  read could still ask -- at %d and %d it has floor p = %.4g, so a variable can be a\n"
              "  real separator and still not be perfect.\n"
              % (n_acc, n_rej, 1 / math.comb(len(table), n_acc)))
        print("  %-14s %-4s %8s %8s %9s %8s  %s"
              % ("variable", "dir", "acc med", "rej med", "U", "p", "perfect"))
        for name, key, direction in (("anneal spread", "spread", "low"),
                                     ("anneal median", "median", "high"),
                                     ("anneal->harden", "step", "high"),
                                     ("binder length", "len", "low")):
            acc = [r[key] for r in table if r["accepted"]]
            rej = [r[key] for r in table if not r["accepted"]]
            flags = [bool(r["accepted"]) for r in table]
            vals = [r[key] for r in table]

            def stat(lab, vals=vals, n_acc=n_acc):
                a = [v for v, f in zip(vals, lab) if f]
                b = [v for v, f in zip(vals, lab) if not f]
                # centre U so the two-sided |.| test is symmetric under the null
                return mann_whitney_u(a, b) - len(a) * len(b) / 2

            obs, pv, total, how, se = perm_p(stat, flags)
            perfect = (max(acc) < min(rej)) if direction == "low" else (min(acc) > max(rej))
            fmt = "%8.0f" if key == "len" else "%8.2f"
            print(("  %-14s %-4s " + fmt + fmt + " %9.1f %8.4f  %s")
                  % (name, direction, statistics.median(acc), statistics.median(rej),
                     obs + n_acc * (len(table) - n_acc) / 2, pv, "yes" if perfect else "no"))
        print("\n  p is %s over %d relabelings." % (how, total))


if __name__ == "__main__":
    main()
