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
import json
import os
import pathlib
import random
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
    out = []
    for c in range(32):
        if c in BLOCKED:
            continue
        f = LEASES / f"{HOST}-card{c}.json"
        if not f.is_file():
            out.append(c)
            continue
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue                      # a half-written lease is not evidence of a free chip
        held = d.get("released") is None or pathlib.Path(f"/proc/{d['pid']}").exists()
        if not held:
            out.append(c)
    return out


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
    return subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)


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
    ap.add_argument("--stop-on-fail", action="store_true")
    a = ap.parse_args()
    logdir = pathlib.Path(a.work)
    logdir.mkdir(parents=True, exist_ok=True)

    for item in a.plan:
        size, _, contig = item.partition(":")
        size = int(size)
        for attempt in range(1, a.tries + 1):
            cards = free_cards()
            if not cards:
                print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {a.model} {size}: "
                      f"no free chip, waiting", flush=True)
                time.sleep(60)
                continue
            # RANDOM, not the first free chip. Three of these drivers run at once against one
            # lease dir, and taking cards[0] made all three pick the same chip every minute:
            # two lose the race, retry, and pick the same one again. Spreading the choice is
            # what turns a retry into progress.
            card = random.choice(cards)
            print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {a.model} {size} "
                  f"attempt {attempt} on card {card}", flush=True)
            p = run_rung(a, size, contig, card)
            blob = (p.stdout or "") + (p.stderr or "")
            print(blob[-1500:], flush=True)
            log = logdir / f"log_{a.model}_{size}.txt"
            ran = CONTENTION not in (log.read_text(errors="replace") if log.is_file() else "")
            if ran:
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
