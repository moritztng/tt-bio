"""Walk one structure model up the cdk2x2 ladder on one Wormhole chip, for size_limits.

perf/whceil/ladder.py does the folding and names the wall. This adds the three things a
published ceiling needs that it does not carry: the AICLK sampled DURING each rung, the load
the rung ran under, and the commit it ran on. It also keeps the chip's lease file claimed
between rungs, because tt_bio's lease is held only by the process that opens the device and
every other row on the box takes a chip whose file reads `released`.

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

HOLDER = "worker:mgx-ceilings"


def _hold(card: str, stop: threading.Event) -> None:
    """Re-claim OUR lease between rungs: only a file this walk's own fold released.

    Never a file another holder wrote. The first draft re-claimed any file whose holder was not
    this row, and on 2026-09-23 it took card 21 from mgx-speed in the gap between two of that
    row's folds. The first fold of a walk takes the chip through tt_bio's own acquire.
    """
    from tt_bio.device_lease import lease_dir, lease_host
    path = os.path.join(lease_dir(), f"{lease_host()}-card{card}.json")
    while not stop.is_set():
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o664)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            stop.wait(5)
            continue
        try:
            try:
                meta = json.loads(os.pread(fd, 4096, 0) or b"{}")
            except ValueError:
                meta = {}
            if meta.get("released") and meta.get("holder") == HOLDER:
                new = {"host": lease_host(), "card": card, "holder": HOLDER, "pid": os.getpid(),
                       "acquired": time.time(), "released": None,
                       "note": "held between folds by a mgx-ceilings walk"}
                os.ftruncate(fd, 0)
                os.pwrite(fd, (json.dumps(new) + "\n").encode(), 0)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
        stop.wait(5)


def _release(card: str) -> None:
    from tt_bio.device_lease import lease_dir, lease_host
    path = os.path.join(lease_dir(), f"{lease_host()}-card{card}.json")
    with open(path, "r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        meta = json.loads(fh.read() or "{}")
        if meta.get("pid") == os.getpid():
            meta["released"] = time.time()
            fh.seek(0)
            fh.truncate()
            fh.write(json.dumps(meta) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--card", required=True)
    ap.add_argument("--sizes", required=True, help="ascending residue counts, csv")
    ap.add_argument("--rung-dir", default="/home/agent/scratch/mgxceil/rungs")
    ap.add_argument("--depth", type=int, default=8192)
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
    from tt_bio.device_lease import lease_dir, lease_host
    try:
        meta = json.loads(Path(lease_dir(), f"{lease_host()}-card{a.card}.json").read_text())
    except (OSError, ValueError):
        meta = {}
    if meta and not meta.get("released") and meta.get("holder") != HOLDER:
        sys.exit(f"card {a.card} is held by {meta.get('holder')}; not taking it")
    stop = threading.Event()
    threading.Thread(target=_hold, args=(a.card, stop), daemon=True).start()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        for n in (int(s) for s in a.sizes.split(",")):
            rung = Path(a.rung_dir) / f"cdk2x2_{n}_d{a.depth}.yaml"
            load0 = os.getloadavg()[0]
            with during(period=10.0) as clk:
                row = run_rung(a.model, rung, int(a.card), Path(a.out_root), a.timeout, env,
                               list(a.extra))
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
    finally:
        stop.set()
        time.sleep(0.5)
        _release(a.card)
    return 0


if __name__ == "__main__":
    sys.exit(main())
