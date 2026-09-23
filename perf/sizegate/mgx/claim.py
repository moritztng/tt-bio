"""Run a list of ladder jobs on whglx, each on the next chip whose lease reads free.

    python claim.py <max chips> <jobs file>

A job line is `<tree> <record|check> <model> [rungs]`; blank lines and `#` comments are skipped.
A chip is taken only when nobody holds its flock, its metadata reads released or names a dead
pid, it has read that way for QUIET_S, and no hold.py of any row is watching it. The first
version used the flock-and-metadata test alone, the one hold.py uses, and on 2026-09-23 it took
chips 2 and 29 from live mgx-diffusion chains and 18 from this row's own check, each in the gap
between two folds. The claim writes this process's pid into the metadata before the chain
starts, and the chain's own hold.py keeps it between folds. A job whose log shows it lost the
chip to another opener is put back in the queue and that chip is skipped for the rest of the run.
Never touches chip 1 or 24-27 (the live app and a co-tenant) or a chip with a cardblock marker.
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
QUIET_S = 60
CONTENDED = b"is in use by"         # tt_bio.device_lease.DeviceInUseError's text


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def watched():
    """Cards some row's hold.py is keeping (argv `hold.py <card> <pid>`), from /proc."""
    cards = set()
    for pid in os.listdir("/proc"):
        try:
            argv = open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")
        except Exception:
            continue
        for i, a in enumerate(argv[:-1]):
            if a.endswith(b"hold.py") and argv[i + 1].isdigit() and int(pid) != os.getpid():
                cards.add(int(argv[i + 1]))
    return cards


def claim(card):
    path = os.path.join(lease_dir(), f"{lease_host()}-card{card}.json")
    try:
        if time.time() - os.path.getmtime(path) < QUIET_S:
            return False
    except OSError:
        pass
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
        if meta.get("released") and time.time() - float(meta["released"]) < QUIET_S:
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
    running, skip = {}, set()                      # card -> (Popen, job, {log: offset})
    while jobs or running:
        for card, (p, job, logs) in list(running.items()):
            if p.poll() is None:
                continue
            del running[card]
            lost = any(CONTENDED in open(f, "rb").read()[off:]
                       for f, off in logs.items() if os.path.exists(f))
            print(f"[{time.strftime('%FT%TZ', time.gmtime())}] card {card} exit {p.returncode}"
                  f"{', lost the chip to another opener: requeued' if lost else ''}", flush=True)
            if lost:
                skip.add(card)
                jobs.append(job)
        if jobs and len(running) < cap:
            busy = watched()
            for card in CARDS:
                if card in running or card in skip or card in busy \
                        or os.path.exists(BLOCK.format(card)) or not claim(card):
                    continue
                job = jobs.pop(0)
                tree, mode, model, *rungs = job
                root = os.path.expanduser(tree)
                logdir = os.path.join(root, "perf/sizegate/mgx/logs")
                logs = {os.path.join(logdir, f): os.path.getsize(os.path.join(logdir, f))
                        for f in os.listdir(logdir) if f.startswith(model + ".")}
                for tag in ("check", "rec-low", "rec-top", f"rec-{''.join(rungs)}"):
                    logs.setdefault(os.path.join(logdir, f"{model}.{tag}.log"), 0)
                print(f"[{time.strftime('%FT%TZ', time.gmtime())}] card {card}: {mode} {model} "
                      f"{' '.join(rungs)} in {tree}", flush=True)
                running[card] = (subprocess.Popen(
                    [os.path.join(root, "perf/sizegate/mgx/ladder_card.sh"),
                     mode, str(card), model, *rungs], stdin=subprocess.DEVNULL), job, logs)
                break
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
