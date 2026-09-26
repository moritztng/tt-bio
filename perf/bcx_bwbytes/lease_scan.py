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
exact `.json` suffix is the live set. An unparseable file is REPORTED, not skipped: this decides
whether to open a device, and the safe reading of a file we cannot read is "occupied".

Prints one line per conflicting lease; prints nothing and exits 0 when the card is unclaimed.
"""
import glob
import json
import os
import sys

SHORT = {"tt-quietbox": "qb1", "tt-quietbox2": "qb2"}


def conflicts(leases, card, host):
    names = {host, SHORT.get(host, host)}
    out = []
    for f in sorted(glob.glob(os.path.join(leases, "*.json"))):
        try:
            d = json.load(open(f))
        except Exception as exc:
            out.append(f"{os.path.basename(f)}: unparseable ({exc}) -- refusing rather than guessing")
            continue
        if str(d.get("card")) != str(card):
            continue
        if str(d.get("host")) in names:
            out.append("{}: holder={} pid={} note={}".format(
                os.path.basename(f), d.get("holder"), d.get("pid"), d.get("note", "")))
    return out


def main():
    leases, card = sys.argv[1], sys.argv[2]
    host = sys.argv[3] if len(sys.argv) > 3 else os.uname().nodename
    for line in conflicts(leases, card, host):
        print(line)


if __name__ == "__main__":
    main()
