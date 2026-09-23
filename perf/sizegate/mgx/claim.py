"""Run a list of ladder jobs on whglx, each on the next chip whose lease reads free.

    python claim.py <max chips> <jobs file>

A job line is `<tree> <record|check> <model> [rungs]`; blank lines and `#` comments are skipped.
A chip is taken only when nobody holds its flock AND its metadata reads released or names a dead
pid, which is the same test hold.py uses, so a live holder is never displaced. The claim writes
this process's pid into the metadata before the chain starts, and the chain's own hold.py keeps it
between folds. Never touches chip 1 or 24-27 (the live app and a co-tenant) or a chip with a
cardblock marker.
"""
import fcntl
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from tt_bio.device_lease import lease_dir, lease_host  # noqa: E402

NEVER = {1, 24, 25, 26, 27}
CARDS = [c for c in range(32) if c not in NEVER]
BLOCK = os.path.expanduser("~/.coworker/state/cardblock-whglx-{}")
HOLDER = "worker:mgx-instrument"
POLL_S = 0.5


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def claim(card):
    path = os.path.join(lease_dir(), f"{lease_host()}-card{card}.json")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return False
    try:
        try:
            meta = json.loads(os.pread(fd, 4096, 0) or b"{}")
        except Exception:
            meta = {}
        if not (meta.get("released") or not alive(meta.get("pid") or 0)):
            return False
        new = {"host": lease_host(), "card": str(card), "holder": HOLDER, "pid": os.getpid(),
               "acquired": time.time(), "released": None,
               "note": "claimed for a size-ladder chain"}
        os.ftruncate(fd, 0)
        os.pwrite(fd, (json.dumps(new) + "\n").encode(), 0)
        return True
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def main():
    cap, jobs = int(sys.argv[1]), []
    for line in open(sys.argv[2]):
        line = line.split("#")[0].split()
        if line:
            jobs.append(line)
    running = {}                                   # card -> Popen
    while jobs or running:
        for card, p in list(running.items()):
            if p.poll() is not None:
                print(f"[{time.strftime('%FT%TZ', time.gmtime())}] card {card} exit "
                      f"{p.returncode}", flush=True)
                del running[card]
        if jobs and len(running) < cap:
            for card in CARDS:
                if card in running or os.path.exists(BLOCK.format(card)) or not claim(card):
                    continue
                tree, mode, model, *rungs = jobs.pop(0)
                print(f"[{time.strftime('%FT%TZ', time.gmtime())}] card {card}: {mode} {model} "
                      f"{' '.join(rungs)} in {tree}", flush=True)
                running[card] = subprocess.Popen(
                    [os.path.join(os.path.expanduser(tree), "perf/sizegate/mgx/ladder_card.sh"),
                     mode, str(card), model, *rungs], stdin=subprocess.DEVNULL)
                break
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
