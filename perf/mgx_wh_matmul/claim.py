#!/usr/bin/env python3
"""Claim the first whglx chip whose lease is idle (released more than 120 s ago) for
worker:mgx-wh-matmul, holding it with HOLD_PID. Polls until one frees; prints the card."""
import fcntl, json, os, sys, time
from pathlib import Path
L = Path.home() / "leases"
BLOCKED = {1, 4, 10, 11, 15, 17, 22, 24, 25, 26, 27, 31}
hold = int(sys.argv[1])
while True:
    for c in range(32):
        if c in BLOCKED:
            continue
        p = L / f"j10glx02-card{c}.json"
        try:
            d = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        # Only a chip its holder released cleanly: a holder that died with the lease open can leave
        # the chip dirty, and card 31's bring-up then hung holding the host's device-open lock.
        idle = d.get("released") and time.time() - float(d["released"]) > 120
        if not idle:
            continue
        with open(p, "a") as fh:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                continue
            p.write_text(json.dumps({"host": "j10glx02", "card": str(c), "holder": "worker:mgx-wh-matmul",
                                     "pid": hold, "acquired": time.time(), "released": None,
                                     "note": "held between probes by mgx-wh-matmul"}))
        print(c, flush=True)
        sys.exit(0)
    time.sleep(10)
