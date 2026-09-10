#!/usr/bin/env python3
"""Re-read every recorded rung's own evidence and correct a verdict that says "wall" where the
evidence says "never got a working card".

Two flavours of that, both recurring: CONTENDED (a co-tenant held the card) and WEDGED (the card
would not come up -- opendde:1536 stalled, left the chip throwing in
risc_firmware_initializer.cpp:1115, and the next TWO rungs were recorded FAIL at 1536 by a card
that never executed an instruction; `tt-smi -r 0` cleared it).

run_rung.py classifies contention at write time now, but rows written before it did are still
in results.jsonl, and one of them (protenix-v1 at 1536, 23:45Z) reads FAIL on a run whose whole
130 s was a lease wait behind a sibling's unpinned pytest. The correction comes from the row's
OWN recorded tail, not from anything remembered, and the original verdict is kept in
`reclassified_from` so the repair is auditable rather than a quiet overwrite.

The classifier is imported from run_rung.py rather than re-typed, so there is one definition of
"a co-tenant had the card" in this harness.

Idempotent. Takes the same lock as run_rung._write and report.py --backfill.
"""
import fcntl
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WT = HERE.parents[1]

_spec = importlib.util.spec_from_file_location("bh1536_run_rung", HERE / "run_rung.py")
_rr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rr)
not_a_measurement = _rr.not_a_measurement


def _evidence(row) -> str:
    """Everything this rung recorded, log first.

    The stored `tail` is not enough on its own: it used to keep only the last 1200 chars, and a
    device-open failure writes its fatal at the TOP of the log, so the tails for opendde-abag and
    protenix-v1 at 1536 held nothing but trailing click frames. The run's own fold.log is still on
    disk, so read that too -- still the run's evidence, not anything remembered.
    """
    parts = [row.get("tail") or ""]
    label = f"{row['model']}_{row['size']}" + (f"_{row['tag']}" if row.get("tag") else "")
    log = HERE / "runs" / label / "fold.log"
    if log.is_file():
        try:
            parts.append(log.read_text(errors="replace"))
        except OSError:
            pass
    return "\n".join(parts)


def main() -> int:
    jl = HERE / "results.jsonl"
    with jl.open("r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in fh if line.strip()]
        changed = []
        for row in rows:
            if row.get("verdict") in ("PASS", "CONTENDED", "WEDGED"):
                continue
            why = not_a_measurement(row.get("exit") or 0, _evidence(row))
            if why:
                changed.append((row["model"], row["size"], row["verdict"], why))
                row["reclassified_from"] = row["verdict"]
                row["verdict"] = why
        fh.seek(0)
        fh.truncate()
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    for model, size, was, why in changed:
        print(f"{model} {size}: {was} -> {why} (its own log says so)")
    print(f"{len(changed)} of {len(rows)} rows corrected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
