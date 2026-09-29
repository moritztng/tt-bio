"""The run's wall-clock cap, as one absolute instant that survives every restart.

Ask 9115 granted this reproduction **5 days of wall clock** on one board pair. Not a step count,
and not 5 days of process uptime, and that second distinction is the whole module. The run
loop's own ``max_seconds`` is measured from ``time.monotonic()`` at process start, and over 5
days qb2 is expected to kill the ranks somewhere between 9 and 47 times (5 watchdog resets in
one day at this load profile, 2026-09-13). A per-process cap therefore caps the longest gap
between two resets and never caps the run: 47 restarts each granted the full budget is 47 times
the grant.

So the deadline is an epoch instant, written into the output directory the first time the run
starts and read by every process after it. A reboot does not move it, a watchdog reset does not
move it, and re-running the launcher with a different ``--days`` does not move it either --
:func:`resolve` refuses to extend a deadline that already exists, because "quietly extend the
run to reach the bar" is the one thing this leg is specifically forbidden to do. Extending is a
decision Moritz takes, and it is taken by deleting the file on purpose.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

__all__ = ["resolve", "remaining", "read"]

NAME = "deadline.json"


def read(out_dir) -> dict | None:
    """The deadline record, or ``None`` if this run has never started."""
    path = Path(out_dir) / NAME
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def resolve(out_dir, days: float | None = None, *, now: float | None = None) -> dict:
    """The run's deadline record, anchored on first call and immutable after it.

    ``days`` is required on the first call and ignored on every later one. Ignored rather than
    honoured because every later call is a restart after a reset, and a restart that re-reads
    ``--days 5`` from its own command line would give the run five fresh days each time.
    """
    out_dir = Path(out_dir)
    found = read(out_dir)
    if found is not None:
        return found
    if days is None:
        raise ValueError(f"no {NAME} in {out_dir} and no --days given: the first launch of a "
                         f"run has to say how long the grant is")
    now = time.time() if now is None else now
    record = {"started": now, "days": float(days), "deadline": now + float(days) * 86400.0,
              "grant": "ask 9115: one whole board pair on qb2, 5 days of wall clock"}
    out_dir.mkdir(parents=True, exist_ok=True)
    # Written whole and then moved, so a reset in the middle of the write cannot leave a
    # truncated deadline that the next launch parses as a shorter grant.
    tmp = out_dir / (NAME + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n")
    tmp.replace(out_dir / NAME)
    return record


def remaining(out_dir, *, now: float | None = None) -> float:
    """Seconds of grant left, never negative. ``0.0`` means the cap has been reached."""
    record = read(out_dir)
    if record is None:
        return 0.0
    now = time.time() if now is None else now
    return max(0.0, record["deadline"] - now)
