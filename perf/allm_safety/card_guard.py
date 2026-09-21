#!/usr/bin/env python3
"""Refuse, at launch, any card this row was not granted -- and hold the lease from second one.

WHY THIS EXISTS. On 2026-09-21 this row fanned a sibling run onto qb1 card 0 with
`TT_BIO_LEASE_CARDS=1,0` while card 0 was `land-standing`'s dispatch grant. It collided twice and
cost that row a whole gate arm. Two separate defects made it possible and both are fixed here.

1. SELF-GRANTING. `TT_BIO_LEASE_CARDS` is the DISPATCHER'S record of what was handed out, and
   `device_lease.granted_cards()` refuses any open outside it -- that is the single choke point
   built to stop exactly this. Writing `1,0` on my own launch line did not widen my entitlement,
   it forged the permission slip the check reads. A worker must never edit that variable.

2. A LATE LEASE. `CardSetLease` is acquired at DEVICE OPEN, which on these harnesses is minutes
   after launch, behind torch import and model load. A probe that finds a card idle at launch says
   nothing about who owns it three minutes later, so "check then launch" is a race by construction
   and re-checking between arms does not close it. The lease has to span the whole launch
   (`card-lease-spans-the-whole-launch-not-the-device-use`).

So: `preflight()` runs as the FIRST statement of a harness, before torch. It refuses a card outside
this row's grant when anyone else holds it, and then acquires the real lease immediately and keeps
it for the process lifetime. A co-tenant launching afterwards is refused cleanly by the same
mechanism instead of racing.

Exit code 75 on refusal: NOT-A-VERDICT, so a refused leg reads as "did not run", never as a result.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

#: This row's own dispatch grant on qb1. Not a knob -- if the dispatcher hands out a different
#: card, that is what `TT_BIO_LEASE_CARDS` will already say and `_declared_grant` picks it up.
ROW_GRANT_DEFAULT = {"1"}
REFUSE_EXIT = 75

_held = []          # keeps the CardSetLease alive for the process lifetime


def _lease_file(card: str) -> Path:
    d = os.environ.get("TT_BIO_LEASE_DIR") or "/home/ttuser/.coworker/state/leases"
    return Path(d) / f"{os.environ.get('TT_BIO_LEASE_HOST', 'tt-quietbox')}-card{card}.json"


def _pid_alive(pid) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def holder_of(card: str):
    """`(holder, pid)` of the live lease on `card`, or None when nobody holds it."""
    f = _lease_file(card)
    if not f.is_file():
        return None
    try:
        d = json.loads(f.read_text())
    except (OSError, ValueError):
        return None
    if d.get("released") is not None:
        return None
    if not _pid_alive(d.get("pid")):
        return None               # flock is kernel-released on death; a stale record is not a hold
    return d.get("holder"), d.get("pid")


def _requested_cards():
    v = os.environ.get("TT_VISIBLE_DEVICES")
    if v is None:
        return None               # unset means the whole box; refuse below
    return {t.strip() for t in v.split(",") if t.strip()}


def preflight(row_grant=None, holder=None, hold=True):
    """Refuse an ungranted card; optionally hold the lease for the whole launch.

    `hold=True` (default) is for a harness that opens the device IN THIS PROCESS: the lease is
    acquired now, before torch and the model load, and handed to `tenstorrent._device_lease` so the
    later open reuses it.

    `hold=False` is for a runner that SPAWNS a child which opens the device. Holding the lease here
    would make the parent a co-tenant of its own child: `tt_bio` refuses the child with "physical
    card N is in use by worker:<me> -- the same holder identity in a DIFFERENT process", which is
    exactly what happened to the first opendde 1024 rung (rc=75, 0 structures, 130 s wasted). The
    child's own acquire is the right owner in that shape -- its lease then spans its launch, which
    is the property that matters -- so this checks the grant and the holder and releases at once.
    `in-process-patch-never-reaches-a-spawn-child`.
    """
    row_grant = set(row_grant or ROW_GRANT_DEFAULT)
    holder = holder or os.environ.get("TT_BIO_LEASE_HOLDER", "worker:allm-safety")

    want = _requested_cards()
    if want is None:
        print("card_guard: TT_VISIBLE_DEVICES is unset, which opens EVERY card on the box. "
              "Refusing.", flush=True)
        sys.exit(REFUSE_EXIT)

    lease_env = {t.strip() for t in os.environ.get("TT_BIO_LEASE_CARDS", "").split(",") if t.strip()}
    if lease_env - row_grant:
        print(f"card_guard: TT_BIO_LEASE_CARDS={sorted(lease_env)} widens this row's grant "
              f"{sorted(row_grant)}. That variable is the dispatcher's record of what was handed "
              f"out, not a knob a worker may set. Refusing.", flush=True)
        sys.exit(REFUSE_EXIT)

    for card in sorted(want - row_grant):
        h = holder_of(card)
        if h:
            print(f"card_guard: card {card} is outside this row's grant {sorted(row_grant)} and is "
                  f"held by {h[0]} (pid {h[1]}). Refusing.", flush=True)
            sys.exit(REFUSE_EXIT)
        print(f"card_guard: card {card} is outside the grant {sorted(row_grant)} and unheld; "
              f"claiming it NOW rather than at device open.", flush=True)

    # Acquire immediately, before the minutes of model loading that made the old path a race.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tt_bio.device_lease import CardSetLease, DeviceInUseError
    try:
        _held.append(CardSetLease(cards=sorted(want)).acquire())
    except DeviceInUseError as e:
        print(f"card_guard: {e}", flush=True)
        sys.exit(REFUSE_EXIT)
    if not hold:
        # Grant and holder are checked; the child owns the device from here.
        release()
        print(f"card_guard: {sorted(want)} granted and unheld by others; released so the spawned "
              f"child can acquire it.", flush=True)
        return None

    # HAND THE LEASE OVER to the module that would otherwise take it again.
    # `tenstorrent.get_device` does `if _device_lease is None: _device_lease = CardSetLease()...`
    # (tenstorrent.py:5285). Without this the guard's flock and tt_bio's acquire are two holders of
    # the same card IN ONE PROCESS, and the second one blocks on the first for the full timeout and
    # then refuses with "in use by worker:allm-safety (pid <me>)" -- the process refusing itself.
    # Measured: that is exactly what the first wired run did. Pre-populating the global makes the
    # later open reuse this lease instead of contending with it, and `release_device` still frees
    # it on the normal path.
    import tt_bio.tenstorrent as _T
    if _T._device_lease is None:
        _T._device_lease = _held[0]
    print(f"card_guard: holding {sorted(want)} as {holder} since "
          f"{time.strftime('%H:%M:%SZ', time.gmtime())}, for the whole launch "
          f"(handed to tenstorrent._device_lease).", flush=True)
    return _held[0]


def release():
    while _held:
        try:
            _held.pop().release()
        except Exception:
            pass
