"""Walk one structure model up the cdk2x2 ladder on one Wormhole chip, for size_limits.

perf/whceil/ladder.py does the folding and names the wall. This adds the three things a
published ceiling needs that it does not carry: the AICLK sampled DURING each rung, the load
the rung ran under, and the commit it ran on. It also holds the chip's lease (the flock, not
only the JSON) between rungs and hands it to its own fold, because tt_bio's lease is held only by
the process that opens the device and every other row on the box takes a chip left free.

    python perf/mgxceil/walk.py --model boltz2 --card 21 --sizes 1024,1152,...,2048 \
        --out walk_boltz2.jsonl --out-root runs/ -- --host_threads 2

Stops at the first non-PASS rung: a ceiling is capped below its FIRST failure.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "perf"))
sys.path.insert(0, str(ROOT / "perf" / "whceil"))

from clocksample import during  # noqa: E402
from ladder import run_rung  # noqa: E402
from tt_bio.device_lease import DeviceInUseError, DeviceLease, lease_dir, lease_host  # noqa: E402

HOLDER = "worker:mgx-ceilings"


def _descendants(root: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for st in Path("/proc").glob("[0-9]*/stat"):
        try:
            f = st.read_text().rsplit(")", 1)[1].split()
        except OSError:
            continue
        kids.setdefault(int(f[1]), []).append(int(st.parent.name))
    out, todo = [], list(kids.get(root, []))
    while todo:
        pid = todo.pop()
        out.append(pid)
        todo += kids.get(pid, [])
    return out


def _flock_holders(path: str) -> set[int]:
    """Pids holding a flock on ``path``, read from /proc/locks."""
    st = os.stat(path)
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    out = set()
    for line in Path("/proc/locks").read_text().splitlines():
        f = line.split()
        if len(f) > 5 and f[1] == "FLOCK" and f[5] == dev:
            out.add(int(f[4]))
    return out


def _handoff(lease, stop: threading.Event, back: list) -> None:
    """Hand the chip to this walk's own fold, then queue to take it back when the fold exits.

    tt_bio's lease is the flock, held only while a fold runs, so a walk that let go between
    rungs lost its chip twice on 2026-09-23 (card 9 to mgx-msa-depth, card 3 to mgx-accuracy):
    another row's open took the flock while the next rung was still importing. The walk holds
    the flock itself and drops it only once a process under it has the lease file open, which
    DeviceLease.acquire does for its whole wait. The gap is one of its 0.25 s polls.

    Dropping it is half the job. Rows waiting on the chip poll every 0.25 s while the fold runs,
    and one of them won the fold's exit twice more the same afternoon (card 5 to mgx-bigalloc,
    card 29 to mgx-diffusion). So once the fold holds the flock, this thread blocks on it: the
    kernel wakes a blocked waiter at unlock, ahead of anyone polling, and the chip comes straight
    back to the walk. The regained lease lands in ``back``.
    """
    path = os.path.realpath(lease.path)
    handed = False
    while not stop.is_set() and not handed:
        for pid in _descendants(os.getpid()):
            try:
                handed = any(os.readlink(f"/proc/{pid}/fd/{fd}") == path
                             for fd in os.listdir(f"/proc/{pid}/fd"))
            except OSError:
                continue
            if handed:
                lease.release()
                break
        else:
            stop.wait(0.05)
    while not stop.is_set():
        if _flock_holders(path) & set(_descendants(os.getpid())):
            fd = os.open(path, os.O_RDWR)
            fcntl.flock(fd, fcntl.LOCK_EX)   # returns the moment the fold exits
            again = DeviceLease(card=lease.card, timeout=0)
            again._fd = fd
            again._write_metadata()
            back.append(again)
            return
        stop.wait(0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--sizes", required=True, help="ascending residue counts, csv")
    ap.add_argument("--rung-dir", default="/home/agent/scratch/mgxceil/rungs")
    ap.add_argument("--depth", type=int, default=8192)
    ap.add_argument("--fixture", help="fold this yaml instead of the tiled rung; --sizes then "
                    "names its token count")
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()

    # The sampler's tt-smi honours this, so index 0 of every sample is the granted chip. This
    # process never opens the device; only the ladder's child does.
    os.environ["TT_VISIBLE_DEVICES"] = a.card
    env = {"TT_BIO_SIZE_LIMIT": "0", "TT_BIO_LEASE_DIR": os.environ["TT_BIO_LEASE_DIR"],
           "TT_BIO_LEASE_HOLDER": HOLDER, **dict(kv.split("=", 1) for kv in a.env)}
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--", "tt_bio"],
                                capture_output=True, text=True).stdout.strip())
    try:
        meta = json.loads(Path(lease_dir(), f"{lease_host()}-card{a.card}.json").read_text())
    except (OSError, ValueError):
        meta = {}
    if meta and not meta.get("released") and meta.get("holder") != HOLDER:
        sys.exit(f"card {a.card} is held by {meta.get('holder')}; not taking it")
    os.environ["TT_BIO_LEASE_HOLDER"] = HOLDER
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lease, wait = None, 0.0   # the first claim must find the chip free
    for n in (int(s) for s in a.sizes.split(",")):
        if lease is None:   # first rung, or a fold that never took the lease
            try:
                lease = DeviceLease(card=a.card, timeout=wait).acquire()
            except DeviceInUseError as e:
                sys.exit(f"lost card {a.card} before {n}: {e}")
        wait = 5.0
        stop, back = threading.Event(), []
        handoff = threading.Thread(target=_handoff, args=(lease, stop, back), daemon=True)
        handoff.start()
        rung = Path(a.fixture or Path(a.rung_dir) / f"cdk2x2_{n}_d{a.depth}.yaml")
        load0 = os.getloadavg()[0]
        try:
            with during(period=10.0) as clk:
                row = run_rung(a.model, rung, int(a.card), Path(a.out_root), a.timeout, env,
                               list(a.extra))
        finally:
            stop.set()
            handoff.join(timeout=30)
            lease.release()   # no-op once handed over
            lease = back[0] if back else None
        s = clk.summary().get(0, {})
        row.update(size=n, depth=a.depth, commit=commit, tt_bio_dirty=dirty,
                   aiclk=s, load_start=round(load0, 1), load_end=round(os.getloadavg()[0], 1),
                   nproc=os.cpu_count(), extra=a.extra,
                   ts=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        if row["verdict"] != "PASS":
            break
    if lease is not None:
        lease.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
