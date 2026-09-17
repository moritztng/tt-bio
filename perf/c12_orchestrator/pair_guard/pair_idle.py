#!/usr/bin/env python3
"""Is my p300c board pair's SIBLING chip idle? Run this before any timing read.

`benchlock` mutually excludes benchlock CALLERS. It cannot see a job that never asked for the lock,
and on a p300c the two chips of one board share a power budget, so an uninstrumented sibling moves
your timing anyway. Root-caused 2026-09-17: `c12-compose-fold` held the lock for a whole pass with
`foreign_folds=0` and still read 0.63x its own A/A floor, because `hall-capacity-800aa` was folding
on the sibling chip. Extending benchlock's poll loop cannot fix it -- a board-power-coupled sibling
need not load the host CPU at all.

    exit 0  sibling idle, your timing read is admissible
    exit 1  SIBLING BUSY -- do not measure; wait it out or DEFER
    exit 2  usage/environment problem

Usage:
    python3 pair_idle.py --card 2            # check the sibling of card 2
    python3 pair_idle.py --card 2 --quiet    # exit status only

qb2 pairing (board serial -> chips): ...4103 = dev0+dev1, ...410D = dev2+dev3. Pairs are
(0,1) and (2,3), i.e. sibling = card XOR 1. That is the dispatch granularity tt-bio already
documents for resets; this script applies it to TIMING, which is the part benchlock misses.
"""
import argparse
import subprocess
import sys
from pathlib import Path

PAIRS = {0: 1, 1: 0, 2: 3, 3: 2}


def holders(card):
    """PIDs holding an fd on /dev/tenstorrent/<card>, via fuser. Empty list if none."""
    node = Path(f"/dev/tenstorrent/{card}")
    if not node.exists():
        print(f"pair_idle: {node} does not exist", file=sys.stderr)
        sys.exit(2)
    for cmd in (["fuser", str(node)], ["sudo", "-n", "fuser", str(node)]):
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            continue
        # fuser writes PIDs to stdout and the "node:" banner to stderr; rc=1 means no holder
        if p.returncode in (0, 1):
            return [int(x) for x in p.stdout.split()]
    print("pair_idle: could not run fuser (needs sudo -n on this host)", file=sys.stderr)
    sys.exit(2)


def describe(pids):
    if not pids:
        return []
    try:
        p = subprocess.run(["ps", "-o", "pid,etime,args", "--no-headers",
                            "-p", ",".join(map(str, pids))],
                           capture_output=True, text=True, timeout=30)
        return [l.strip()[:120] for l in p.stdout.splitlines() if l.strip()]
    except (OSError, subprocess.TimeoutExpired):
        return [f"pid {x} (ps unavailable)" for x in pids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", type=int, required=True, choices=sorted(PAIRS))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    sib = PAIRS[a.card]
    mine, theirs = holders(a.card), holders(sib)

    if not a.quiet:
        print(f"my card {a.card}: {mine or 'no fd'}")
        for l in describe(mine):
            print(f"    {l}")
        print(f"sibling {sib}: {theirs or 'no fd'}")
        for l in describe(theirs):
            print(f"    {l}")

    if theirs:
        if not a.quiet:
            print(f"\nSIBLING BUSY -- card {sib} shares a board power budget with card {a.card}.")
            print("Your timing read is contaminated whether or not you hold benchlock.")
            print("Wait it out or DEFER. Do not measure and explain it afterwards.")
        return 1
    if not a.quiet:
        print(f"\nsibling {sib} idle -- timing read admissible. "
              f"Record BOTH chips' occupancy beside every number.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
