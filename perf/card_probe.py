#!/usr/bin/env python3
"""Which chips on this host can actually be opened, one short subprocess per card.

A chip a crashed process left mid-initialised does not report itself anywhere: `tt-smi` is happy,
the lease file is released, `lsof` shows no holder, and the card looks free. The next open then
either throws `Device 0: Timeout (10000 ms) waiting for physical cores to finish` followed by
`failed to initialize FW`, or hangs in `futex` forever with 130 threads parked, which at the
Python level is indistinguishable from a live job. On a 32-chip galaxy shared by ten workstreams
that is a trap you fall into once per pass: five of the seven idle chips this probe was written
for were in that state on 2026-09-12, and the three fanned-out jobs that landed on them looked
like they were loading a model for ten minutes.

Each card is probed in its OWN subprocess under a timeout, because the failure modes include a
hang and one device context per process is the rule anyway. `HANG` and `DEAD` are both unusable;
the difference is only whether the chip throws or parks.

  python3 perf/card_probe.py                 # every card the driver exposes
  python3 perf/card_probe.py 5,17,18-23      # a list and/or ranges

One thing this deliberately does NOT do is reset anything. `tt-smi -r` on a Galaxy resets more
than the card named, so which dead chip is worth reclaiming is the caller's decision, not this
script's.
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CHILD = r"""
import os, sys
from tt_bio import tenstorrent as T
d = T.get_device()
g = d.compute_with_storage_grid_size()
print("OPEN_OK %s %dx%d" % (d.arch(), g.x, g.y))
"""


def parse_cards(spec: str) -> list[int]:
    if not spec:
        return sorted(int(p.name) for p in Path("/dev/tenstorrent").iterdir() if p.name.isdigit())
    out: list[int] = []
    for part in spec.replace(" ", ",").split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def probe(card: int, timeout: float, holder: str) -> dict:
    env = dict(os.environ,
               TT_VISIBLE_DEVICES=str(card), TT_BIO_LEASE_CARDS=str(card),
               TT_BIO_LEASE_HOLDER=holder)
    t0 = time.perf_counter()
    try:
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, timeout=timeout,
                           capture_output=True, text=True)
        blob = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return {"card": card, "state": "HANG", "s": round(timeout, 1),
                "detail": f"no result in {timeout:.0f}s"}
    dt = round(time.perf_counter() - t0, 1)
    if "OPEN_OK" in blob:
        line = next(l for l in blob.splitlines() if "OPEN_OK" in l)
        return {"card": card, "state": "OK", "s": dt, "detail": line.split("OPEN_OK ", 1)[1]}
    for needle, state in (("failed to initialize FW", "DEAD"),
                          ("waiting for physical cores to finish", "DEAD"),
                          ("DeviceInUseError", "BUSY")):
        if needle in blob:
            return {"card": card, "state": state, "s": dt, "detail": needle}
    tail = [l for l in blob.strip().splitlines() if l.strip()][-1:] or [""]
    return {"card": card, "state": "ERROR", "s": dt, "detail": tail[0][:160]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cards", nargs="?", default="", help="e.g. 5,17,18-23; default every card")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--jobs", type=int, default=4, help="cards probed at once")
    ap.add_argument("--holder", default=f"card_probe:{os.getpid()}")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    cards = parse_cards(args.cards)
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        rows = sorted(ex.map(lambda c: probe(c, args.timeout, args.holder), cards),
                      key=lambda r: r["card"])
    for r in rows:
        print(f"card {r['card']:3d}  {r['state']:5s}  {r['s']:6.1f}s  {r['detail']}")
    by = {}
    for r in rows:
        by.setdefault(r["state"], []).append(r["card"])
    print("\n" + "  ".join(f"{k}={sorted(v)}" for k, v in sorted(by.items())))
    if args.json:
        args.json.write_text(json.dumps({"rows": rows, "by_state": by}, indent=1))
    return 0 if by.get("OK") else 1


if __name__ == "__main__":
    raise SystemExit(main())
