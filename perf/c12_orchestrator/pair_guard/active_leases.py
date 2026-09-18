#!/usr/bin/env python3
"""Count leases that are ACTIVELY HELD: released is null AND the recorded pid is alive.

The reboot watcher used `find -mmin -5` on this directory as its "is a worker using a card"
guard. That is wrong: these files are written on device ACQUIRE and rewritten on RELEASE, so the
mtime tracks device open/close churn, not worker liveness. Measured 2026-09-17: card2's mtime sat
frozen across a 40 s window with `"released": null` while c12-compose-fold was actively working
between folds. Any row doing a wheel build, an analysis phase or anything else off-device for more
than 5 minutes reads as idle and could be rebooted out from under.

`released is null` alone would be too sticky the other way -- a crashed holder leaves it null
forever -- so the pid must also be alive.
"""
import json, os, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "/home/ttuser/.coworker/state/leases")
# A missing or unreadable lease dir must NOT read as "nobody holds a card". The caller guards a
# REBOOT on this number, so the only safe failure is a loud one: exit non-zero and let the caller
# substitute a large value. Returning 0 here would say the box is idle when we simply cannot see.
if not d.is_dir():
    print(f"active_leases: {d} is not a directory", file=sys.stderr)
    sys.exit(2)
active = []
for f in sorted(d.glob("*.json")):
    try:
        j = json.loads(f.read_text())
    except (ValueError, OSError):
        continue
    if j.get("released") is not None:
        continue
    pid = j.get("pid")
    if not isinstance(pid, int):
        continue
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        continue          # crashed holder, released never written -- not a live hold
    except PermissionError:
        pass              # alive, owned by another user
    active.append(f"{f.stem}:{j.get('holder','?')}:pid{pid}")
if "-v" in sys.argv:
    for a in active:
        print(a, file=sys.stderr)
print(len(active))
