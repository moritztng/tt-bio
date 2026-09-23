#!/usr/bin/env python3
"""Who is on which chip, by all four signals at once.

Run this before concluding that a row is starved rather than that its probe is too strict.
The four signals disagree routinely on this box and the table is how you see which one is
lying: on 2026-09-23 card 31's lease read `released` and named `worker:mgx-msa-depth` while
`worker:mgx-bigalloc` held its device node open, and cards 14 and 18 read stale with two fds
open on each.

    python3 perf/mgxscale/cards.py
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from perf.mgxscale.job import (BLOCKED, HOST, LEASES, _live, free_cards,  # noqa: E402
                               held_cards, node_of, open_cards, pinned_cards)


def main() -> int:
    op, pin, held, free = open_cards(), pinned_cards(), held_cards(), set(free_cards())
    print(f"host {HOST}   leases {LEASES}")
    print("%5s%6s%8s%8s%7s%7s  %s" % ("card", "node", "fd", "pinned", "held", "free", "lease"))
    for c in range(32):
        if c in BLOCKED:
            print("%5d%6d%8s%8s%7s%7s  cardblocked" % (c, node_of(c), "-", "-", "-", "-"))
            continue
        try:
            rec = json.loads((LEASES / f"{HOST}-card{c}.json").read_text())
            state = ("held/live" if rec.get("released") is None and _live(rec.get("pid"))
                     else "stale" if rec.get("released") is None else "released")
            who = f"{state} {rec.get('holder', '?')}"
        except Exception:
            who = "none"
        print("%5d%6d%8s%8s%7s%7s  %s"
              % (c, node_of(c), "yes" if c in op else "", "yes" if c in pin else "",
                 "yes" if c in held else "", "YES" if c in free else "", who))
    print(f"\nopen {len(op)}  pinned {len(pin)}  held {len(held)}  "
          f"FREE {sorted(free)} ({len(free)} of {32 - len(BLOCKED)} usable)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
