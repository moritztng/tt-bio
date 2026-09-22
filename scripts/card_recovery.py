"""Hand a card back to the driver after a killed fold, before anything opens it again.

A fold that WEDGES gets its process group killed on a timeout. The kill reaches the
multiprocessing child that holds `/dev/tenstorrent/N`, but the card is left in a state the
driver cannot re-initialise, and the NEXT device open is what kills the host:

    tenstorrent 0000:01:00.0: Failed to set initial power state: -5     (-5 is EIO)

repeated every few seconds until the kernel log stops mid-session with no shutdown sequence.
qb2 died that way three times on 2026-09-22 (14:36:52Z, 17:20:15Z, 20:45:36Z); the last two log
the error against the PCI address of the card the killed fold was on.

So a wedge-kill is only half of the recovery. The other half is this module, and it is not
SIGINT-before-SIGTERM: a wedged fold spins inside ttnn C++, where a Python signal handler is
never scheduled between bytecodes, so the polite signal is ignored by exactly the process that
needs it.

`tt-smi -r N` on a p300c resets the whole BOARD PAIR, not the chip, so a blind reset here would
take a co-tenant's job down with it. When the sibling is busy this REFUSES, and a refusal is a
result the caller must respect by stopping: opening a dirty card is what kills the host, and a
dead host costs the co-tenant far more than a wait.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

#: p300c board pairs. `tt-smi -r` on either member resets both
#: (`qb2-tt-smi-reset-resets-board-pair-not-chip`).
BOARD_PAIRS = ((0, 1), (2, 3))

OK, REFUSED, FAILED = "ok", "refused", "failed"


def board_sibling(card: int) -> int | None:
    """The other chip on this card's board, whose job a reset would also kill."""
    for a, b in BOARD_PAIRS:
        if card == a:
            return b
        if card == b:
            return a
    return None


def fd_holders(card: int) -> list[int]:
    """Pids holding an fd on /dev/tenstorrent/<card>, by /proc scan rather than lsof.

    No sudo and no external binary, which matters because this runs on a recovery path where
    the host may already be unhealthy. It only sees this user's processes; on a single-user box
    that is every process that can hold the card.
    """
    want = f"/dev/tenstorrent/{card}"
    held = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                if os.readlink(fd) == want:
                    held.append(int(proc.name))
                    break
        except (OSError, PermissionError):
            continue
    return held


def _tt_smi() -> str | None:
    found = shutil.which("tt-smi")
    if found:
        return found
    for path in (Path.home() / ".local/bin/tt-smi", Path("/usr/local/bin/tt-smi")):
        if path.exists():
            return str(path)
    return None


def reset_after_kill(card: int, *, timeout_s: float = 180.0, say=print) -> str:
    """Reset `card` after a killed fold. Returns OK, REFUSED or FAILED.

    REFUSED is not a soft failure: the card is still dirty, so the caller must not open it.
    """
    card = int(card)
    sib = board_sibling(card)
    if sib is not None:
        busy = fd_holders(sib)
        if busy:
            say(f"[card-recovery] REFUSING to reset card {card}: `tt-smi -r` resets the whole "
                f"board pair and sibling card {sib} is held by pid(s) {busy}. The card is still "
                f"dirty -- do NOT open it.")
            return REFUSED
    smi = _tt_smi()
    if not smi:
        say(f"[card-recovery] no tt-smi on PATH; cannot reset card {card}")
        return FAILED
    say(f"[card-recovery] resetting card {card} (board pair {card}/{sib}) after a wedge-kill")
    try:
        rc = subprocess.run([smi, "-r", str(card)], timeout=timeout_s,
                            stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT).returncode
    except subprocess.TimeoutExpired:
        say(f"[card-recovery] tt-smi -r {card} did not return in {timeout_s:.0f}s")
        return FAILED
    if rc != 0:
        say(f"[card-recovery] tt-smi -r {card} exited {rc}")
        return FAILED
    still = fd_holders(card)
    if still:
        say(f"[card-recovery] card {card} reset but pid(s) {still} still hold an fd")
        return FAILED
    say(f"[card-recovery] card {card} reset, no fd holders")
    return OK


def visible_card(default: int = 0) -> int:
    """The single card this process was granted, from TT_VISIBLE_DEVICES."""
    vis = (os.environ.get("TT_VISIBLE_DEVICES") or "").strip()
    first = vis.split(",")[0].strip() if vis else ""
    return int(first) if first.isdigit() else default
