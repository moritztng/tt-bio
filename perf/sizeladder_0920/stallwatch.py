#!/usr/bin/env python3
"""Kill a wedged size-ladder fold, scoped to the model that wedged. One process, all cards.

WHY THIS REPLACES THE IN-SCRIPT WATCHDOG. drive2.sh's watchdog detected openfold3's 768 aa wedge
correctly and then killed nothing for twenty minutes, because it looked for the fold with

    pgrep -f "tt_bio.main predict <work dir>/"

and the fold's argv names its FIXTURE, `perf/size512/fixtures/cdk2x2_768.yaml`, not the work dir.
The pattern matched zero processes and a zero match is indistinguishable from a healthy box
(`guard-goes-quiet-when-its-subject-outgrows-its-word-list`). So this logs a MISS loudly when it
finds a stall and no process to blame it on, instead of returning quietly.

And the pattern would have been wrong even if it had matched: every model's folds run out of one
shared `--keep` work dir, so a stall on card 2 would have SIGKILLed the healthy ladders on cards
0, 1 and 3. Scope here is the census process whose `--label <model>-<rung>-<rep>` names the model
that stalled, and its descendants.

Innermost first. Killing the census parent leaves the predict running and still holding the card
lease -- that happened on card 0 at 12:32Z, where an orphaned spawn worker at 127 % CPU kept the
lease and made the next two record attempts die at device open
(`fleet-kill-outer-pid-leaves-orphan-engine`).

SIGKILL rather than SIGINT: the only signal observed to clear this wedge.

Usage:  stallwatch.py [--stall 300] [--poll 30] [--work DIR]
"""
import argparse
import os
import re
import subprocess
import sys
import time

FOLD = re.compile(r"^(?P<model>.+)-(?P<rung>\d+)-(?:rep\d+|warmup)\.log$")


def stamp():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def pgrep(pattern):
    out = subprocess.run(["pgrep", "-f", "--", pattern], capture_output=True, text=True).stdout
    return [int(x) for x in out.split()]


def descendants(pid):
    """pid and everything under it, deepest last so the caller can kill in reverse."""
    order, frontier = [pid], [pid]
    while frontier:
        nxt = []
        for p in frontier:
            kids = subprocess.run(["pgrep", "-P", str(p)], capture_output=True,
                                  text=True).stdout.split()
            nxt += [int(k) for k in kids]
        order += nxt
        frontier = nxt
    return order


def newest_per_model(work):
    out = {}
    for name in os.listdir(work):
        m = FOLD.match(name)
        if not m:
            continue
        p = os.path.join(work, name)
        try:
            mt = os.path.getmtime(p)
        except OSError:
            continue
        if mt > out.get(m["model"], (0, ""))[0]:
            out[m["model"]] = (mt, name)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stall", type=int, default=300)
    ap.add_argument("--poll", type=int, default=30)
    ap.add_argument("--work", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sizegate", "work"))
    args = ap.parse_args()
    print(f"{stamp()} stallwatch on {args.work}, stall={args.stall}s", flush=True)
    killed_at = {}
    while True:
        time.sleep(args.poll)
        now = time.time()
        for model, (mt, name) in newest_per_model(args.work).items():
            quiet = now - mt
            if quiet < args.stall:
                continue
            # A model whose ladder simply finished also goes quiet. Only act while a census for
            # THIS model is actually running.
            census = pgrep(f"--label {model}-")
            if not census:
                continue
            if now - killed_at.get(model, 0) < args.stall:
                continue
            tree = []
            for c in census:
                tree += descendants(c)
            if not tree:
                print(f"{stamp()} MISS {model}: {name} quiet {quiet:.0f}s and a census is "
                      f"running, but no process tree matched -- NOT a healthy box, fix the "
                      f"pattern", flush=True)
                continue
            print(f"{stamp()} STALL {model}: {name} quiet {quiet:.0f}s -- SIGKILL {tree[::-1]}",
                  flush=True)
            for p in reversed(tree):
                try:
                    os.kill(p, 9)
                except OSError:
                    pass
            killed_at[model] = now


if __name__ == "__main__":
    sys.exit(main())
