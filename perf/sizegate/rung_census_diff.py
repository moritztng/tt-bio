#!/usr/bin/env python3
"""Which levers change behaviour between two rungs of a size ladder.

The size-ladder baseline records a lever census per rung, but it records each rung
on its own: reading it tells you what fired at 768, not that something which fired
at 512 stopped firing at 768. A lever that goes dark inside a size band is the
defect this campaign hunts (`one-size-tuning-is-a-standing-defect-class`), and it
is only visible in the diff.

Usage:
    rung_census_diff.py census_rf3-256-rep0.json census_rf3-512-rep0.json ...

Rungs are taken from the `label` field (`<model>-<rung>-<rep>`) so the files can be
listed in any order. Prints one row per lever whose served/declined pattern is not
the same shape at every rung, and stays silent about the levers that behave.
"""
import json
import re
import sys


def _rung(census: dict, path: str) -> int:
    m = re.search(r"-(\d+)-(?:rep\d+|warmup)$", str(census.get("label") or ""))
    if not m:
        raise SystemExit(f"{path}: label {census.get('label')!r} has no -<rung>- part")
    return int(m.group(1))


def _reject(census: dict, path: str) -> str | None:
    """Why this census cannot be diffed, or None if it can.

    A fold that died before it imported tt_bio still writes a census, and every row
    in it reads `absent`. Diffed against a good rung that is 30-odd levers all going
    dark at once -- a whole screen of findings, none of them real. Measured on the
    640 warmup this campaign killed during a co-tenant cleanup: rc=-15, 0 processes,
    32/32 rows absent. Refuse it by the fold's own exit status rather than by the
    all-absent shape, so a model that genuinely imports nothing is still diffable."""
    rc = census.get("rc")
    if rc not in (0, None):
        return f"fold exited {rc}"
    procs = census.get("processes")
    if isinstance(procs, int):
        procs_n = procs
    else:
        procs_n = len(procs or ())
    if not procs_n:
        return "no process dumped counters"
    return None


def _shape(row: dict) -> str:
    """What a lever DID at this rung, coarse enough that call-count growth with the
    sequence length does not read as a change. Only the served/declined pattern is
    the signal: an off flag, an on flag that never fired, one that fired and was
    never declined, and one that was declined at least once are four different
    states and the transitions between them are what a size band hides."""
    if row.get("served") is None and row.get("declined") is None:
        return "absent"          # module not imported by this model
    if str(row.get("resolved")) in ("False", "None"):
        return "off"             # flag off by configuration, not by size
    served, declined = int(row.get("served") or 0), int(row.get("declined") or 0)
    if served == 0 and declined == 0:
        return "unused"          # on, but this model never reaches the call site
    if served == 0:
        return "DARK"            # on, reached, and declined every time
    return "served" if declined == 0 else "mixed"


def main(paths: list[str]) -> int:
    by_rung: dict[int, dict[str, dict]] = {}
    for p in paths:
        with open(p) as fh:
            c = json.load(fh)
        bad = _reject(c, p)
        if bad:
            print(f"skipping {p}: {bad}", file=sys.stderr)
            continue
        by_rung.setdefault(_rung(c, p), {}).update(
            {r["flag"]: r for r in c.get("rows") or []})
    rungs = sorted(by_rung)
    if len(rungs) < 2:
        raise SystemExit("need at least two rungs to diff")

    flags = sorted({f for rows in by_rung.values() for f in rows})
    width = max(len(f) for f in flags)
    print(f"rungs: {', '.join(map(str, rungs))}")
    header = "  ".join(f"{n:>8}" for n in rungs)
    print(f"{'lever':<{width}}  {header}")
    changed = 0
    for flag in flags:
        shapes = [_shape(by_rung[n].get(flag, {})) for n in rungs]
        if len(set(shapes)) == 1:
            continue
        changed += 1
        print(f"{flag:<{width}}  " + "  ".join(f"{s:>8}" for s in shapes))
    if not changed:
        print("\nno lever changes shape across these rungs.")
    else:
        print(f"\n{changed} lever(s) change shape. A `served` -> `DARK`/`unused` "
              "transition is the one to root-cause: the lever stopped serving because "
              "of the size, not because of the model.")
    # Served counts for the levers that stay served everywhere are still worth a look
    # when an exponent jumps: a lever that keeps serving but serves a shrinking FRACTION
    # of its call sites reads as `served` here and still costs the wall.
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
