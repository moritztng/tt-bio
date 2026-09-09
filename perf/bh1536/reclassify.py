#!/usr/bin/env python3
"""Re-read every recorded rung's own tail and correct a verdict that says "wall" where the
evidence says "never got the card".

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
contended = _rr.contended


def main() -> int:
    jl = HERE / "results.jsonl"
    with jl.open("r+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        rows = [json.loads(line) for line in fh if line.strip()]
        changed = []
        for row in rows:
            if row.get("verdict") in ("PASS", "CONTENDED"):
                continue
            if contended(row.get("exit") or 0, row.get("tail") or ""):
                changed.append((row["model"], row["size"], row["verdict"]))
                row["reclassified_from"] = row["verdict"]
                row["verdict"] = "CONTENDED"
        fh.seek(0)
        fh.truncate()
        for row in rows:
            fh.write(json.dumps(row) + "\n")
        fcntl.flock(fh, fcntl.LOCK_UN)
    for model, size, was in changed:
        print(f"{model} {size}: {was} -> CONTENDED (its own tail says the card was held)")
    print(f"{len(changed)} of {len(rows)} rows corrected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
