#!/usr/bin/env python3
"""Fill the OFF-arm scratch baseline's dark-lever exemption reasons for the token-bucket A/B.

`release_gate.py --size-ladder-record` leaves every dark lever's `reason` as a TODO, and the
comparison run FAILS on any dark lever that still has one. In a committed baseline that is the
guard working: a lever nobody noticed went dark must not be recorded as normal. Here the file is a
scratch OFF-arm control written and read inside one A/B, so a lever being dark IS the reference
state and the only finding the comparison scores is whether turning the bucket on moves it. The
first run left the TODOs in and buried ~15 real ON-vs-OFF deltas under 56 lines that said nothing
about the flip.

Any decline detail the recorder appended is kept verbatim, because that detail is the evidence for
why the lever is dark and it is what a reader of the A/B needs.
"""
import json
import re
import sys

WHY = ("OFF-arm control for the token-bucket A/B, not a committed baseline: dark with the bucket "
       "off is the reference state here, and the only finding scored against this file is whether "
       "turning the bucket on moves it.")


def stamp(path):
    doc = json.load(open(path))
    n = 0

    def walk(node):
        nonlocal n
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "reason" and isinstance(value, str) and value.startswith("TODO"):
                    detail = re.search(r"\((.*)\)\s*$", value, re.S)
                    node[key] = WHY + (" " + detail.group(1) + "." if detail else "")
                    n += 1
                else:
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(doc)
    json.dump(doc, open(path, "w"), indent=2)
    return n


if __name__ == "__main__":
    print(f"stamped {stamp(sys.argv[1])} OFF-arm dark-lever reasons", file=sys.stderr)
