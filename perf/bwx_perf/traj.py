#!/usr/bin/env python3
"""What one BindCraft 2 trajectory costs, from a campaign that ran to its own stop condition.

    traj.py <campaign dir> [--chips 32] [--accept-rate 0.226]

The round is not the unit anybody buys. A campaign's own `rounds.json` (written by
`perf/bcx_p10_campaign/campaign_run.py`, one line per gradient round per slot) plus the wall
between its first and last round gives the two figures that matter: seconds of ONE CHIP per
completed trajectory, and trajectories an hour on a whole Galaxy.

`--accept-rate` prices a design rather than a trajectory. It is a property of BindCraft 2's
settings and the target, not of the board, so it is an argument with the measured Blackhole rate
as its default (7 accepted of 31 completed trajectories, `state/perf10/bcx-p10-campaign.md`)
rather than something this script pretends to measure from two trajectories.
"""
import argparse
import json
import pathlib
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("campaign", help="a campaign out/ dir holding rounds.json")
    ap.add_argument("--chips", type=int, default=32, help="chips on the Galaxy")
    ap.add_argument("--accept-rate", type=float, default=7 / 31)
    args = ap.parse_args()

    rows = json.loads((pathlib.Path(args.campaign) / "rounds.json").read_text())
    slots: dict[str, list] = {}
    for r in rows:
        slots.setdefault(r["slot"], []).append(r["t"])
    wall = max(r["t"] for r in rows) - min(r["t"] for r in rows)
    gaps = [b - a for ts in slots.values() for a, b in zip(ts, ts[1:])]
    per_traj = wall / len(slots)
    an_hour = 3600.0 / per_traj

    print(f"{len(slots)} trajectories interleaved on one chip, {len(rows)} gradient rounds, "
          f"{wall:.0f} s wall")
    for slot, ts in sorted(slots.items()):
        print(f"  {slot}: {len(ts)} rounds, {ts[-1] - ts[0]:.0f} s from its first to its last")
    print(f"round, amortised over the slots:      {statistics.median(gaps) / len(slots):.2f} s")
    print(f"chip-seconds per completed trajectory {per_traj:.0f} s")
    print(f"trajectories an hour, one chip        {an_hour:.2f}")
    print(f"trajectories an hour, {args.chips:2d} chips        {an_hour * args.chips:.0f}")
    print(f"chip-hours per accepted design        {per_traj / args.accept_rate / 3600:.2f} "
          f"(at {args.accept_rate:.3f} accepted per completed trajectory)")


if __name__ == "__main__":
    main()
