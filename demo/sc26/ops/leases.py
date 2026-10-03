"""Fleet lease files for the demo's chips, so the dispatcher never grants one to another row.

    leases.py hold 0,1,2,3 PID prints the chips now held for the demo (a comma list)
    leases.py release 0,1,2,3

The files are the ones tt_bio's device lease and the fleet's pick_card read:
~/.coworker/state/leases/<host>-card<N>.json. `hold` writes each with PID as both pid and
detached_pid (the service's main process, which then execs the server), so the card
reads busy for as long as the service lives, including between two worker starts. A chip whose
lease names another holder that is still alive is left alone and not used: that is somebody
else's work. `release` stamps "released" only on files the demo wrote.
"""
import json
import os
import socket
import sys
import time
from pathlib import Path

HOLDER = os.environ.get("TT_BIO_LEASE_HOLDER", "worker:sc26-demo")
DIR = Path(os.environ.get("TT_BIO_LEASE_DIR", Path.home() / ".coworker/state/leases"))
HOST = os.environ.get("TT_BIO_LEASE_HOST") or socket.gethostname()


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True


def path(c):
    return DIR / f"{HOST}-card{c}.json"


def read(c):
    try:
        return json.loads(path(c).read_text())
    except (OSError, ValueError):
        return None


def foreign(m):
    """A lease that names somebody else and is still held."""
    if not m or m.get("holder") == HOLDER:
        return False
    return alive(m.get("pid")) or alive(m.get("detached_pid"))


def hold(chips, pid):
    DIR.mkdir(parents=True, exist_ok=True)
    held = []
    for c in chips:
        m = read(c)
        if foreign(m):
            print(f"chip {c}: held by {m.get('holder')} (pid {m.get('pid')}), not used", file=sys.stderr)
            continue
        if m and m.get("holder") != HOLDER:
            path(c).rename(path(c).with_name(path(c).name + f".released-{time.strftime('%Y-%m-%d')}-dead-holder-sc26-demo-takes-it"))
        tmp = path(c).with_suffix(".tmp")
        tmp.write_text(json.dumps({"host": HOST, "card": str(c), "holder": HOLDER, "pid": pid,
                                   "detached_pid": pid, "acquired": time.time(), "released": None}) + "\n")
        os.replace(tmp, path(c))
        held.append(c)
    return held


def release(chips):
    for c in chips:
        m = read(c)
        if m and m.get("holder") == HOLDER and not m.get("released"):
            m["released"] = time.time()
            path(c).write_text(json.dumps(m) + "\n")


def main():
    cmd, chips = sys.argv[1], [c for c in sys.argv[2].split(",") if c.strip()]
    if cmd == "hold":
        print(",".join(hold(chips, int(sys.argv[3]))))
    elif cmd == "release":
        release(chips)
    else:
        sys.exit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
