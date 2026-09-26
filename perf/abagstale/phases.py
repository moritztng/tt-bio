#!/usr/bin/env python3
"""Did the fold keep moving, or did it stop?

collect.py reports the longest gap between two engine log lines. That number alone cannot
tell a stall from a phase that is simply expensive at this size: the 1536 leg's 697 s window
looks alarming next to the 1024 leg's 197 s until you see both sit in the SAME phase, the one
between the last trunk iteration and diffusion step 0.

So: print the trunk cadence, the trunk-to-diffusion transition and the diffusion loop
separately, per leg. A stall shows up as one phase that stops scaling with the others and as
a missing step count; a cubic phase shows up as a bigger number in every leg at once.
"""
import datetime
import re
import sys
from pathlib import Path

STEP = re.compile(r"^(\d\d:\d\d:\d\d)  \[[^\]]+\]\s+(trunk|diffusion) (\d+)/(\d+)", re.M)


def phases(log: Path) -> str:
    ev = STEP.findall(log.read_text(errors="replace"))
    if not ev:
        return f"{log.stem:8s} no trunk/diffusion markers"
    # The engine stamps time of day with no date, so walk the stamps onto a monotonic line.
    t, day, prev = [], 0, None
    for stamp, *_ in ev:
        cur = datetime.datetime.strptime(stamp, "%H:%M:%S")
        if prev is not None and cur < prev:
            day += 1
        prev = cur
        t.append(cur + datetime.timedelta(days=day))
    tr = [t[i] for i, e in enumerate(ev) if e[1] == "trunk"]
    di = [t[i] for i, e in enumerate(ev) if e[1] == "diffusion"]
    want_tr = int(ev[0][3])
    want_di = int(next(e[3] for e in ev if e[1] == "diffusion")) if di else 0
    cad = sorted((b - a).total_seconds() for a, b in zip(tr, tr[1:]))
    miss = []
    if len(tr) != want_tr:
        miss.append(f"trunk {len(tr)}/{want_tr}")
    if di and len(di) != want_di:
        miss.append(f"diffusion {len(di)}/{want_di}")
    return (f"{log.stem:8s} trunk cadence med {cad[len(cad) // 2]:5.0f} s "
            f"(min {cad[0]:.0f}, max {cad[-1]:.0f}) | "
            f"last trunk -> diffusion 0 {(di[0] - tr[-1]).total_seconds():5.0f} s | "
            f"diffusion loop {(di[-1] - di[0]).total_seconds():5.0f} s | "
            + ("steps all present" if not miss else "MISSING " + ", ".join(miss)))


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    for tag in sys.argv[2:]:
        print(phases(root / "logs" / f"{tag}.log"))
