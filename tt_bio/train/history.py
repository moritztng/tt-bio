"""Reading the run's ``history-rank<N>.jsonl`` back, including the lines a crash tore.

``abb3_run`` appends one JSON object per completed step, line-buffered. Two things happen to
that file that a plain ``json.loads`` per line does not survive, and both were seen on this run
rather than imagined:

* **a SIGKILL lands mid-write** and leaves a truncated last line. Dropping it is right. It is
  one step, and refusing to read the history because the run died the way it was expected to
  die breaks the reader exactly when it is needed.
* **a host reset lands after ext4 journalled the append's size but not its data.** The line
  comes back as a run of NUL bytes followed by the complete record, because the next process
  appended into the same block. On 2026-09-19 that was line 694 of both rank files: 699 NULs
  then a whole step-684 row, digest ``b6b7d48dd9f4fe12617f63d8392e9aca``. Stripping the NULs
  recovers it exactly.

The second case is why this is a module and not three lines in one script. Dropping the torn
line costs more than the step it holds: the reader then sees the counter jump from 692 straight
to 685, and the resume detector reads that as a run that never came off its checkpoint. That is
what happened, on an append-only record, so a healthy run carried a latched INCIDENT. An
instrument that cries wolf once teaches its reader to ignore it.

The repair belongs in the reader and never in the file. ``history-rank*.jsonl`` is the run's own
append-only record, and hand-editing it destroys the evidence that the crash happened.

Stdlib only and no torch, so every reader of the curve can share one parser: the heartbeat that
runs between passes and the curve writer that runs at the cap.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["read_rows"]


def read_rows(path) -> list:
    """Every parseable step record in ``path``, in file order, NUL-torn lines recovered.

    Unparseable lines are dropped rather than raised on, and the count of them is recoverable
    by the caller as ``len(lines) - len(rows)`` if it cares. Nothing here repairs the file.
    """
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.replace("\x00", "").strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
