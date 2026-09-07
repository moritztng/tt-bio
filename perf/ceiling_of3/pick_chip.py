#!/usr/bin/env python3
"""Print one free UMD card index for a measurement on the shared Galaxy, or nothing.

Free means all three of:

* the card is not inside the JapanFold production pool's own ``--device_ids`` grant, read
  from the live worker's argv so the candidate set shrinks by itself the moment the pool
  grows and this never has to be told twice;
* nothing holds the card's device lease -- tested with the same non-blocking ``flock`` the
  lease itself uses, so it is the authoritative answer and not a guess from the metadata;
* nothing holds an fd on the ``/dev/tenstorrent`` node the card actually maps to.

The third check is the one that was wrong. A lease and a ``TT_VISIBLE_DEVICES`` pin both
name a UMD card index, while ``lsof`` and the fd-level collision live on a ``/dev`` node,
and on this Galaxy those two numberings are a permutation with no fixed point at all
(``tt_bio.runtime.tt_bdf_to_index``, measured 2026-09-02). Checking node N and then opening
card N therefore tested the freeness of a chip it never touched: it lost the device race
five times in a row on the first ladder attempt, at two minutes a loss, and every rung it
did run was on a chip whose occupancy had never been read.

    python3 pick_chip.py            # one card, descending, or exit 1
    python3 pick_chip.py --all      # every free card, for the record
"""
from __future__ import annotations

import argparse
import fcntl
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from tt_bio.device_lease import lease_dir, lease_host          # noqa: E402
from tt_bio.runtime import umd_index_to_dev_node               # noqa: E402


#: Last pool grant this picker ever saw, so a momentarily invisible pool cannot widen the
#: candidate set. Beside the leases, which is already the fleet's device-state directory.
POOL_MEMO = os.path.join(os.path.dirname(lease_dir()), "of3ceiling-pool-grant.json")

#: A ``tt-bio worker`` command line and its card grant. Anchored on the subcommand and the
#: flag as REAL TOKENS: a substring test also matches this script's own diagnostics, and the
#: first version crashed on ``grep -- "--device_ids"`` in an unrelated shell line.
WORKER_RE = re.compile(r"(?:^|/)tt-bio\s+worker\s.*?--device_ids[=\s]+([0-9,]+)")


def scan_pool_cards() -> set[int] | None:
    """Cards the production pool granted itself, from the live ``tt-bio worker`` argv.

    ``None`` when no worker line is visible at all, which is NOT the same as an empty grant.
    """
    out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    cards: set[int] = set()
    seen = False
    for line in out.splitlines():
        m = WORKER_RE.search(line)
        if not m:
            continue
        seen = True
        cards |= {int(t) for t in m.group(1).split(",") if t.strip().isdigit()}
    return cards if seen else None


def pool_cards() -> set[int]:
    """Cards this measurement must never take, remembered across pool restarts.

    The pool's worker respawns, and during that window no ``tt-bio worker`` line exists. A
    live-only reading then returns an empty grant and every production chip looks takeable --
    which is exactly what happened at 23:33:28Z, when this offered card 24 while the pool was
    mid-restart. The device lease refused the open, so nothing ran on it, but the lease is the
    last line of defence and should not be the first.

    So the answer is the UNION of what is visible now and every grant seen before, persisted.
    A restart can only ever shrink the candidate set, never widen it. If nothing is visible and
    nothing was remembered, this refuses to answer rather than declaring the box quiet: on a
    host known to run a pool, invisible is not absent.
    """
    live = scan_pool_cards()
    try:
        with open(POOL_MEMO) as f:
            memo = set(json.load(f))
    except Exception:
        memo = set()
    if live is None and not memo:
        raise SystemExit(
            "no `tt-bio worker --device_ids` line and no remembered grant: refusing to pick a "
            "chip, because a pool that is invisible for a moment is not a pool that is absent."
        )
    cards = memo | (live or set())
    if cards != memo:
        os.makedirs(os.path.dirname(POOL_MEMO), exist_ok=True)
        tmp = POOL_MEMO + f".{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(sorted(cards), f)
        os.replace(tmp, POOL_MEMO)
    return cards


def lease_free(card: int) -> bool:
    """True when the card's lease can be flocked right now.

    The flock IS the lease (the json body is observability), and the kernel drops it on any
    process death, so this never reports a crashed holder as live.
    """
    path = os.path.join(lease_dir(), f"{lease_host()}-card{card}.json")
    if not os.path.exists(path):
        return True
    try:
        fd = os.open(path, os.O_RDWR)
    except OSError:
        return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except OSError:
        return False
    finally:
        os.close(fd)


def node_free(node: int) -> bool:
    """True when no process holds an fd on ``/dev/tenstorrent/<node>``.

    ``lsof`` needs root to see another user's fds, and a sudo that cannot run
    non-interactively would silently report every chip free -- which is the one wrong answer
    that matters. Probe sudo once and refuse to answer at all if it is unavailable.
    """
    r = subprocess.run(["sudo", "-n", "lsof", "-t", f"/dev/tenstorrent/{node}"],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):
        raise SystemExit(f"cannot read fds on /dev/tenstorrent/{node}: {r.stderr.strip()}")
    return not r.stdout.strip()


def free_cards() -> list[int]:
    to_node = umd_index_to_dev_node()
    pool = pool_cards()
    return [c for c in sorted(to_node, reverse=True)
            if c not in pool and lease_free(c) and node_free(to_node[c])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--map", action="store_true", help="print the card -> node map and exit")
    a = ap.parse_args()
    to_node = umd_index_to_dev_node()
    if a.map:
        print(" ".join(f"{c}->{to_node[c]}" for c in sorted(to_node)))
        return
    cards = free_cards()
    if not cards:
        raise SystemExit(1)
    print(" ".join(str(c) for c in cards) if a.all else cards[0])


if __name__ == "__main__":
    main()
