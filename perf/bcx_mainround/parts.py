#!/usr/bin/env python3
"""Per (trajectory, round) digest parts (pred, grad, loss) of two arms, side by side.

    parts.py out/a1 out/b1
"""
import json
import sys


def load(d):
    ev = json.load(open(f"{d}/round_events.json"))
    ev = ev.get("events", ev) if isinstance(ev, dict) else ev
    out, seen = {}, {}
    for e in ev:
        if e.get("kind") != "digest":
            continue
        t = e.get("trajectory", e.get("slot", e.get("name")))
        seen[t] = seen.get(t, 0) + 1
        out[(str(t), seen[t])] = e.get("parts", {})
    return out


a, b = load(sys.argv[1]), load(sys.argv[2])
for k in sorted(set(a) | set(b)):
    pa, pb = a.get(k, {}), b.get(k, {})
    print(k, " ".join(f"{p}={'=' if pa.get(p) == pb.get(p) else 'DIFF'}"
                      for p in ("pred", "grad", "loss")))
