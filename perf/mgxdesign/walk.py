#!/usr/bin/env python3
"""Walk a design model's size axis on a CONTENDED Galaxy, one rung per free chip.

`perf/bhdesign/ladder.py` walks the axis and is unchanged by this: it assumes it owns a card
for the length of the walk, which is true on a quiet box and false on whglx, where five MGX
rows fan out over 27 chips and a chip frees and refills between two rungs. There, the ladder
reads a contention refusal as a rung FAILURE -- the run never reached the model, so the rung
measured the fleet, not the ceiling, and a `--stop-on-fail` walk stops at a size that works.

So this driver owns the card choice and the retry, and the ladder still owns the measurement:

  * a free chip is picked PER RUNG out of the device-lease dir tt-bio itself arbitrates on
    (`TT_BIO_LEASE_DIR`, `~/leases` on whglx), never out of the fleet's own lease view, which
    is a different directory and was stale by ten cards when this was written;
  * a rung refused for contention is RETRIED on another chip, never recorded;
  * the cardblocked chips are refused here as well as in `pick_card`, because this driver does
    not go through the fleet's picker at all.

    python3 perf/mgxdesign/walk.py --model rfd3 --plan 512:A1-412,100 1024:A1-924,100 ...
"""
import argparse
import fcntl
import json
import os
import pathlib
import random
import signal
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEASES = pathlib.Path(os.environ.get("TT_BIO_LEASE_DIR", "/tmp/tt-bio-device-leases"))
HOST = os.uname().nodename
# app.japanfold.com (24-27) and the tri_mech co-tenant (1). The fleet blocks these in
# state/cardblock-whglx-*; this driver never reads that dir, so it carries them itself.
BLOCKED = {1, 24, 25, 26, 27}
CONTENTION = "device contention, nothing ran"


def free_cards() -> list[int]:
    """The chips whose lease flock can be taken right now.

    THE FLOCK IS THE LEASE, not the JSON beside it. `tt_bio/device_lease.py` says so in its
    own docstring -- "a dead holder's lock is simply gone, so the next acquire succeeds
    immediately. There is no pid-liveness scan to get wrong" -- and the metadata is written
    INSIDE the lock, so a holder that took a SIGKILL never gets to set `released` and its
    record says `"released": null` forever. Reading that JSON, as the first version of this
    function did, marks every crashed row's chip as permanently held: it saw 6 free chips on a
    box that was cycling rungs constantly, and all three walks sat in "no free chip, waiting".

    Probing the lock is the same test the engine will apply a second later, so a chip that
    passes here is one the rung can really open. The probe releases immediately, so this
    races with other leasers by design -- losing that race is what the contention retry is
    for, and it is a far cheaper error than never trying at all.
    """
    out = []
    for c in range(32):
        if c in BLOCKED:
            continue
        f = LEASES / f"{HOST}-card{c}.json"
        try:
            fd = os.open(str(f), os.O_RDWR | os.O_CREAT, 0o664)
        except OSError:
            continue        # not ours to open (a lease dir owned by another user)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
            out.append(c)
        except OSError:
            pass            # a live holder owns it
        finally:
            os.close(fd)
    return out


# The rung currently running, so a signal can reach it. Killing this driver alone leaves the
# ladder child and its tt_bio.main grandchild ALIVE, holding a chip: on 2026-09-23 a killed
# boltzgen driver left an orphan (ppid 1) computing on card 17 for 18 minutes, and two more
# orphans wrote FAIL rows into the JSONL for rungs their driver no longer owned. A restart of
# this driver must not cost the box a chip.
_CHILD: subprocess.Popen | None = None


def _bail(signum, _frame):
    if _CHILD and _CHILD.poll() is None:
        try:
            os.killpg(os.getpgid(_CHILD.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            _CHILD.wait(timeout=30)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(_CHILD.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
    sys.exit(128 + signum)


def run_rung(a, size: int, contig: str, card: int) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-u", str(ROOT / "perf/bhdesign/ladder.py"),
           "--model", a.model, "--sizes", str(size), "--card", str(card),
           "--target", a.target, "--binder", str(a.binder), "--designs", str(a.designs),
           "--holder", a.holder, "--arch", "wormhole_b0", "--board", "tt-galaxy-wh-l",
           "--out", a.out, "--work", a.work, "--timeout", str(a.timeout)]
    if a.steps:
        cmd += ["--steps", str(a.steps)]
    if contig:
        cmd += ["--rfd3-contig", contig]
    global _CHILD
    # Its own process group, so one killpg reaches the ladder AND the tt_bio.main it spawns.
    _CHILD = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, start_new_session=True)
    out, _ = _CHILD.communicate()
    rc = _CHILD.returncode
    _CHILD = None
    return subprocess.CompletedProcess(cmd, rc, out, "")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--plan", nargs="+", required=True,
                    help="rungs as SIZE or SIZE:CONTIG, walked in the order given")
    ap.add_argument("--target", default="perf/bhdesign/targets/big_1831.cif")
    ap.add_argument("--binder", type=int, default=80)
    ap.add_argument("--designs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--holder", default="worker:mgx-design-ceiling")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--timeout", type=int, default=2400)
    ap.add_argument("--tries", type=int, default=30, help="contention retries per rung")
    ap.add_argument("--stick-s", type=int, default=900,
                    help="seconds to wait for the chip this walk already used before moving to "
                         "another one. Keeps a comparison on one chip, which the speed bar "
                         "requires; past it, measuring on a second chip beats measuring nothing")
    ap.add_argument("--stop-on-fail", action="store_true")
    a = ap.parse_args()
    for _sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(_sig, _bail)
    logdir = pathlib.Path(a.work)
    logdir.mkdir(parents=True, exist_ok=True)

    # The chip the walk has been using. docs/speed-bar.md judges a rung against fit rungs from
    # ONE chip -- `identity` differing across rungs returns VOID, not a pass or a fail -- so a
    # driver that takes whatever is free would produce a ladder no bar can read. It therefore
    # WAITS for the chip it already used, up to `--stick-s`, and only then moves. Moving is still
    # allowed: a walk that stalls forever measures nothing at all, and a rung that had to move
    # says so in its own row, which is what lets the bar void just that comparison.
    stick: int | None = None
    for item in a.plan:
        size, _, contig = item.partition(":")
        size = int(size)
        for attempt in range(1, a.tries + 1):
            # Waiting for a chip is not a retry. Every one of the 27 unblocked chips can be busy
            # for half an hour on this box, and charging that to the retry budget made a walk
            # give up on a saturated Galaxy without ever having run a rung.
            waited = 0
            while not (cards := free_cards()):
                if waited % 600 == 0:
                    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {a.model} {size}: "
                          f"no free chip, waiting ({waited // 60} min so far)", flush=True)
                time.sleep(60)
                waited += 60
            # RANDOM, not the first free chip. Three of these drivers run at once against one
            # lease dir, and taking cards[0] made all three pick the same chip every minute:
            # two lose the race, retry, and pick the same one again. Spreading the choice is
            # what turns a retry into progress.
            if stick is not None and stick not in cards:
                waited_stick = 0
                while waited_stick < a.stick_s and stick not in free_cards():
                    time.sleep(20)
                    waited_stick += 20
                cards = free_cards() or cards
            card = stick if (stick is not None and stick in cards) else random.choice(cards)
            print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {a.model} {size} "
                  f"attempt {attempt} on card {card}", flush=True)
            p = run_rung(a, size, contig, card)
            blob = (p.stdout or "") + (p.stderr or "")
            print(blob[-1500:], flush=True)
            log = logdir / f"log_{a.model}_{size}.txt"
            ran = CONTENTION not in (log.read_text(errors="replace") if log.is_file() else "")
            if ran:
                stick = card
                break
            # The rung never reached the model. Drop the row the ladder just appended, or the
            # jsonl carries a FAIL at a size that was never tried.
            out = pathlib.Path(a.out)
            if out.is_file():
                rows = [l for l in out.read_text().splitlines() if l.strip()]
                if rows and json.loads(rows[-1])["size"] == size:
                    out.write_text("\n".join(rows[:-1]) + ("\n" if rows[:-1] else ""))
            time.sleep(45)
        else:
            print(f"{a.model} {size}: gave up after {a.tries} contention retries", flush=True)
            return 1
        last = json.loads(pathlib.Path(a.out).read_text().splitlines()[-1])
        if last["verdict"] != "PASS" and a.stop_on_fail:
            print(f"STOP: {a.model} {size} {last['verdict']} ({last['mechanism']})", flush=True)
            break
    print(f"{a.model} WALK COMPLETE {time.strftime('%FT%TZ', time.gmtime())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
