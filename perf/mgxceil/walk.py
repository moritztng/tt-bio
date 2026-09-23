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


def _handoff(lease, stop: threading.Event) -> None:
    """Hand the chip to this walk's own fold and to nobody else.

    tt_bio's lease is the flock, held only while a fold runs, so a walk that let go between
    rungs lost its chip twice on 2026-09-23 (card 9 to mgx-msa-depth, card 3 to mgx-accuracy):
    another row's open took the flock while the next rung was still importing. The walk now
    holds the flock itself and drops it only once a process under it has the lease file open,
    which DeviceLease.acquire does for its whole wait. The gap is one of its 0.25 s polls.
    """
    path = os.path.realpath(lease.path)
    while not stop.is_set():
        for pid in _descendants(os.getpid()):
            try:
                fds = os.listdir(f"/proc/{pid}/fd")
            except OSError:
                continue
            for fd in fds:
                try:
                    if os.readlink(f"/proc/{pid}/fd/{fd}") == path:
                        lease.release()
                        return
                except OSError:
                    continue
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
    from tt_bio.device_lease import DeviceInUseError, DeviceLease, lease_dir, lease_host
    try:
        meta = json.loads(Path(lease_dir(), f"{lease_host()}-card{a.card}.json").read_text())
    except (OSError, ValueError):
        meta = {}
    if meta and not meta.get("released") and meta.get("holder") != HOLDER:
        sys.exit(f"card {a.card} is held by {meta.get('holder')}; not taking it")
    os.environ["TT_BIO_LEASE_HOLDER"] = HOLDER
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    wait = 0.0   # the first claim must find the chip free; later ones only cover a fold's exit
    for n in (int(s) for s in a.sizes.split(",")):
        try:
            lease = DeviceLease(card=a.card, timeout=wait).acquire()
        except DeviceInUseError as e:
            sys.exit(f"lost card {a.card} before {n}: {e}")
        wait = 5.0
        stop = threading.Event()
        handoff = threading.Thread(target=_handoff, args=(lease, stop), daemon=True)
        handoff.start()
        rung = Path(a.fixture or Path(a.rung_dir) / f"cdk2x2_{n}_d{a.depth}.yaml")
        load0 = os.getloadavg()[0]
        try:
            with during(period=10.0) as clk:
                row = run_rung(a.model, rung, int(a.card), Path(a.out_root), a.timeout, env,
                               list(a.extra))
        finally:
            stop.set()
            handoff.join()
            lease.release()
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
