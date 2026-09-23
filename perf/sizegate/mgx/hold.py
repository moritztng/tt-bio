"""Keep a chip visibly claimed between the folds of one ladder chain.

tt_bio's lease is a flock taken by the process that opens the device, and a ladder chain is
dozens of short processes. Between two of them the lease file reads `released`, and a picker
that reads the file (the fleet's and every worker's) takes the chip: on 2026-09-23 two MGX rows
took chips 7 and 9 from this row's openbind and rf3 chains between folds. A 5 s poll still lost
chip 15 to mgx-accuracy between esmfold2's 896 and 1024 rungs (07:59Z), 84 minutes of record
gone, so the gap is watched every 0.2 s.

This rewrites the metadata as held by the chain's own pid whenever the flock is FREE and the
file says released. It never writes while anyone holds the flock, so it cannot overwrite a live
holder, and tt_bio's own acquire still decides who opens the device. Exits with the chain.

    python hold.py <card> <chain pid>
"""
import fcntl
import json
import os
import sys
import time

card, chain = sys.argv[1], int(sys.argv[2])
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tt_bio.device_lease import lease_dir, lease_host  # noqa: E402

path = os.path.join(lease_dir(), f"{lease_host()}-card{card}.json")
POLL_S = 0.2
holder = os.environ.get("TT_BIO_LEASE_HOLDER", "worker:mgx-instrument")


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


while alive(chain):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)          # someone is on the device right now; theirs to describe
        time.sleep(POLL_S)
        continue
    try:
        try:
            meta = json.loads(os.pread(fd, 4096, 0) or b"{}")
        except Exception:
            meta = {}
        if meta.get("released") or not alive(meta.get("pid") or 0):
            new = {"host": lease_host(), "card": card, "holder": holder, "pid": chain,
                   "acquired": time.time(), "released": None,
                   "note": "held between folds by a size-ladder chain"}
            os.ftruncate(fd, 0)
            os.pwrite(fd, (json.dumps(new) + "\n").encode(), 0)
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
    time.sleep(POLL_S)
