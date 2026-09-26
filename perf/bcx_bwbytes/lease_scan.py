#!/usr/bin/env python3
"""Every LIVE lease on this host that names a given card, under either naming convention.

Two conventions are live on these boxes at once and they are not redundant.
`<hostname>-cardN.json` is written by the engine's own device-open path and is PID-KEYED; it can
read released 2.1 s after acquire, because a trajectory closes the card once per leg, and a row
that CYCLES arms drops it entirely in the gap between them. `<short>-cardN.json` is the
orchestrator's occupancy guard, kept deliberately BECAUSE of that gap: it holds for 60 s after
the device node empties. Reading only the first is being blind to the guard that covers precisely
the window in which a taker would collide, and that window produced two near-misses in one night.

A released lease is RENAMED to `*.json.released-...` on this fleet rather than deleted, so an
exact `.json` suffix is the candidate set.

**A lease whose pid is provably dead is STALE, not a conflict**, and this file learned that the
hard way: refusing on existence alone deadlocked the first real claim this row ever made. qb1
card 2 freed at 03:35Z, `card_free.sh` said FREE because it judges a lease by pid liveness, and
the chain refused on `tt-quietbox-card2.json` naming pid 187751, which had exited. Two gates
disagreeing about what "leased" means is worse than either gate alone, so this one now matches
the one that decides whether to claim at all. Stale leases are still REPORTED, on stderr, because
a card freed by a leak is worth knowing about.

Everything else is a conflict, including a lease with no pid and a file that will not parse: this
decides whether to open a device, and the safe reading of a lease we cannot evaluate is held.
"""
import glob
import json
import os
import sys

SHORT = {"tt-quietbox": "qb1", "tt-quietbox2": "qb2"}


def _alive(pid):
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError) as exc:
        return isinstance(exc, PermissionError)     # another user's live process is still live
    return True


def scan(leases, card, host):
    """(conflicts, stale) -- conflicts block a claim, stale ones are reported and do not."""
    names = {host, SHORT.get(host, host)}
    conflicts, stale = [], []
    for f in sorted(glob.glob(os.path.join(leases, "*.json"))):
        base = os.path.basename(f)
        try:
            d = json.load(open(f))
        except Exception as exc:
            conflicts.append(f"{base}: unparseable ({exc}) -- refusing rather than guessing")
            continue
        if str(d.get("card")) != str(card) or str(d.get("host")) not in names:
            continue
        pid = d.get("pid")
        line = "{}: holder={} pid={} note={}".format(base, d.get("holder"), pid, d.get("note", ""))
        if pid is None:
            conflicts.append(line + " -- no pid, cannot be shown stale")
        elif _alive(pid):
            conflicts.append(line)
        else:
            stale.append(line + " -- pid is dead, treating as stale")
    return conflicts, stale


def conflicts(leases, card, host):
    return scan(leases, card, host)[0]


def main():
    leases, card = sys.argv[1], sys.argv[2]
    host = sys.argv[3] if len(sys.argv) > 3 else os.uname().nodename
    bad, stale = scan(leases, card, host)
    for line in stale:
        print(line, file=sys.stderr)
    for line in bad:
        print(line)


if __name__ == "__main__":
    main()
