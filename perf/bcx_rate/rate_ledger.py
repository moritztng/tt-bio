#!/usr/bin/env python3
"""BindCraft 2 acceptance ledger: accepted designs per COMPLETED trajectory, and what a
trajectory costs.

Why this exists. On 2026-09-25 the BCX campaign published "5,137 chip-seconds per accepted
design, 19.4x the BoltzGen funnel" and handed it to the competition campaign for credit sizing.
5,137 s was the wall time of the one trajectory that accepted; five completed rejections were
sitting on the same box and were not in the denominator. A BindCraft 2 rejection is not cheap --
it comes at a gradient stage, after most of the gradient work -- so leaving them out understated
the figure by 7.1x. This script divides by every completed trajectory, always, and refuses to
report a rate without its interval.

Run it against one or more arm project directories (the `--out` of `run_arm.py`):

    python3 perf/bcx_rate/rate_ledger.py ~/bcx_accept_art/accept_s4 ~/bcx_mutate_art/route_s3_arm

Each directory supplies `.campaign_state.json` for the accounting, `1_Trajectories/
!_Trajectories.csv` for the termination stage per attempt, `arm_stamp.json` for the
configuration, and the trajectory subdirectory mtimes for the wall boundaries. Attempts still in
flight are excluded: an attempt with no verdict is not a rejection.
"""
import argparse
import csv
import json
import sys
from math import comb
from pathlib import Path

BOLTZGEN_CHIP_S_PER_DESIGN = 264.3  # state/cmp/LOAD.md, whglx at AICLK 1000 sampled during


def _beta_cdf(x, a, b):
    """Regularized incomplete beta for integer a, b."""
    n = a + b - 1
    return sum(comb(n, j) * x ** j * (1 - x) ** (n - j) for j in range(a, n + 1))


def _beta_inv(p, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided interval for k successes in n trials."""
    lo = 0.0 if k == 0 else _beta_inv(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_inv(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def read_arm(project):
    """One arm's completed trajectories, in the order the run produced them."""
    project = Path(project)
    stamp = json.loads((project / "arm_stamp.json").read_text())
    state = json.loads((project / ".campaign_state.json").read_text())

    rows = list(csv.DictReader(open(project / "1_Trajectories" / "!_Trajectories.csv")))
    ends = {}
    for design_dir in (project / "1_Trajectories").iterdir():
        if design_dir.is_dir():
            files = [f.stat().st_mtime for f in design_dir.rglob("*") if f.is_file()]
            if files:
                ends[design_dir.name] = max(files)

    start = _parse_utc(stamp["started_utc"])
    trajectories = []
    previous = start
    for row in sorted((r for r in rows if r["design"] in ends), key=lambda r: ends[r["design"]]):
        end = ends[row["design"]]
        trajectories.append({
            "arm": project.name,
            "seed": stamp["seed"],
            "design": row["design"],
            "length": int(row.get("length") or 0),
            # BindCraft 2 leaves `terminated` empty for a trajectory that ran the whole way.
            "terminated": row["terminated"] or "completed",
            "accepted": not row["terminated"],
            "chip_s": end - previous,
        })
        previous = end

    # An arm that has finished keeps working after its last trajectory directory stops growing:
    # the MPNN validation ensemble, the refolds and the ranking all land in 2_Refolded/ and
    # 3_Ranked/. That tail is part of what the accepted design cost and the chip was held for it,
    # so it belongs to the last trajectory. An arm with an attempt still in flight owns its tail
    # for that attempt instead, and gets nothing added.
    in_flight = len(state["attempted"]) - len(trajectories)
    if trajectories and in_flight == 0:
        tail = max(f.stat().st_mtime for f in project.rglob("*") if f.is_file())
        if tail > previous:
            trajectories[-1]["chip_s"] += tail - previous

    return {
        "project": project,
        "stamp": stamp,
        "state": state,
        "trajectories": trajectories,
        # The arm's own count is authoritative for acceptance; the CSV only says where each
        # attempt stopped. They agree unless a trajectory completed and then failed validation.
        "accepted": state["accepted"],
        "in_flight": in_flight,
    }


def _parse_utc(text):
    from datetime import datetime, timezone
    return datetime.strptime(text, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("projects", nargs="+", help="arm project directories")
    parser.add_argument("--reference", type=float, default=BOLTZGEN_CHIP_S_PER_DESIGN,
                        help="chip-s per design to compare against (default: BoltzGen funnel)")
    args = parser.parse_args(argv)

    arms = [read_arm(p) for p in args.projects]
    trajectories = [t for arm in arms for t in arm["trajectories"]]
    if not trajectories:
        sys.exit("no completed trajectories in any project -- nothing to divide by")

    print(f"{'arm':<16}{'seed':>5}  {'draw':<6}{'terminated':<12}{'chip-s':>9}")
    for t in trajectories:
        print(f"{t['arm']:<16}{t['seed']:>5}  l{t['length']:<5}"
              f"{('ACCEPTED' if t['accepted'] else t['terminated']):<12}{t['chip_s']:>9,.0f}")

    n = len(trajectories)
    accepted = sum(arm["accepted"] for arm in arms)
    total = sum(t["chip_s"] for t in trajectories)
    in_flight = sum(arm["in_flight"] for arm in arms)
    print(f"\n{n} completed trajectories, {accepted} accepted, {total:,.0f} chip-s"
          f"{f' ({in_flight} in flight, excluded)' if in_flight else ''}")

    if accepted == 0:
        lo, hi = clopper_pearson(0, n)
        print(f"\nrate 0 / {n}, 95% CI {lo:.4f} - {hi:.4f}")
        print(f"cost per accepted design is unbounded above; at the interval's upper rate it is "
              f"at least {total / n / hi:,.0f} chip-s")
        return

    rate = accepted / n
    lo, hi = clopper_pearson(accepted, n)
    mean = total / n
    cost = mean / rate
    print(f"\nrate                 {accepted}/{n} = {rate:.3f} per trajectory"
          f"   95% CI {lo:.4f} - {hi:.4f}")
    print(f"mean trajectory      {mean:,.0f} chip-s")
    print(f"per accepted design  {cost:,.0f} chip-s"
          f"   95% CI {mean / hi:,.0f} - {mean / lo:,.0f}")
    print(f"against reference    {cost / args.reference:.1f}x"
          f"   95% CI {mean / hi / args.reference:.1f}x - {mean / lo / args.reference:,.0f}x")
    print(f"designs/h/chip       {3600 / cost:.4f}")

    if n < 10:
        print("\nAll trajectories here are pre-autotune. BindCraft 2 reviews and nudges its own "
              "stage thresholds\nevery AUTOTUNE_REVIEW_TRAJECTORIES = 10 attempts, so a ledger "
              "that crosses ten must be split there.")


if __name__ == "__main__":
    main()
