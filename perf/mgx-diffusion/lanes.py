"""Keep up to N plans folding, each on a whglx chip that is genuinely free.

    python perf/mgx-diffusion/lanes.py <max lanes> <plan> [<plan> ...]

A chip is free when nobody holds its lease flock, its lease file is released or names a dead
pid, and it is not one of the cardblocked chips (the live JapanFold app and a co-tenant). A plan
runs through launch.sh on one chip; a plan whose chain exits 0 is finished, and one that lost
its chip to another row between folds (ladder.py exits non-zero) is started again on the next
free chip, resuming from runs.jsonl. Exits when every plan has finished.
"""
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEASES = Path(os.environ.get("TT_BIO_LEASE_DIR", Path.home() / "leases"))
HOST = "j10glx02"
BLOCKED = {1, 24, 25, 26, 27}          # state/cardblock-whglx-*: never touched
CARDS = [c for c in range(32) if c not in BLOCKED]


def alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:
        return False


def free(card):
    path = LEASES / f"{HOST}-card{card}.json"
    if not path.exists():
        return False                   # never leased here: not a chip we know is idle
    with open(path) as fp:
        try:
            fcntl.flock(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return False
        try:
            meta = json.loads(fp.read() or "{}")
        except ValueError:
            meta = {}
        finally:
            fcntl.flock(fp, fcntl.LOCK_UN)
    return bool(meta.get("released")) or not alive(meta.get("pid") or 0)


class Adopted:
    """A launch.sh chain this supervisor did not start (it was running before it)."""

    def __init__(self, pid):
        self.pid = pid

    def poll(self):
        return None if alive(self.pid) else 0


def adopt(plans):
    """Chains already running for these plans, so a restart never starts a second copy."""
    found = {}
    for d in Path("/proc").iterdir():
        try:
            argv = (d / "cmdline").read_bytes().split(b"\0")
        except (OSError, ValueError):
            continue
        a = [x.decode(errors="replace") for x in argv if x]
        if len(a) >= 4 and a[0] == "bash" and a[1].endswith("launch.sh"):
            for plan in plans:
                if Path(a[3]).name == Path(plan).name:
                    found[plan] = (Adopted(int(d.name)), int(a[2]))
    return found


def main():
    n, plans = int(sys.argv[1]), sys.argv[2:]
    running, done = adopt(plans), set()
    log = HERE / "logs" / "lanes.log"
    log.parent.mkdir(exist_ok=True)
    say = lambda m: open(log, "a").write(f"[{time.strftime('%FT%TZ', time.gmtime())}] {m}\n")
    for plan, (_p, card) in running.items():
        say(f"{plan} already running on {card}, adopted")
    while len(done) < len(plans):
        for plan, (proc, card) in list(running.items()):
            if proc.poll() is not None:
                rc = subprocess.run(["tail", "-1", str(HERE / "logs" / f"{Path(plan).stem}-{card}.log")],
                                    capture_output=True, text=True).stdout
                del running[plan]
                if "EXIT 0" in rc:
                    done.add(plan)
                say(f"{plan} on {card} ended: {rc.strip()}")
        for plan in plans:
            if plan in done or plan in running or len(running) >= n:
                continue
            busy = {c for _p, c in running.values()}
            card = next((c for c in CARDS if c not in busy and free(c)), None)
            if card is None:
                break
            running[plan] = (subprocess.Popen(["bash", str(HERE / "launch.sh"), str(card), plan],
                                              start_new_session=True), card)
            say(f"{plan} started on {card}")
            time.sleep(20)             # let its first fold take the lease before the next pick
        time.sleep(60)
    say("all plans finished")


if __name__ == "__main__":
    main()
