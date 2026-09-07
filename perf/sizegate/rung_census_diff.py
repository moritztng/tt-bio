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


def _served(row: dict) -> int | None:
    """How many call sites this lever actually served, or None if it was absent."""
    if row.get("served") is None and row.get("declined") is None:
        return None
    return int(row.get("served") or 0)


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
    header = "  ".join(f"{n:>13}" for n in rungs)
    print(f"{'lever':<{width}}  {header}")
    lost, shape_only = [], []
    for flag in flags:
        cells = [(_shape(by_rung[n].get(flag, {})), _served(by_rung[n].get(flag, {})))
                 for n in rungs]
        shapes = [c[0] for c in cells]
        counts = [c[1] for c in cells]
        # A shape change alone is not a finding. TRIATT_PERSISTENT_MASK reads
        # `served` -> `mixed` between 512 and 640 aa on Wormhole with its served count
        # flat at 1088: `_PM_OVER_L1` latches a refused (Sq, Sk, q_chunk, k_chunk,
        # kv_factor) config, the L1 ladder drops to a narrower chunk and that one
        # serves, so every call site is still served and the declines are the probes
        # it cost to get there. What actually costs the wall is a lever serving FEWER
        # call sites than it did a rung earlier, which is `TRANSPOSE_L1_RESIDENT`
        # (1088 -> 40 at 640 aa, its pair tensor outgrowing aggregate L1). Report the
        # served counts and split the two, rather than suppressing either: the shape
        # change is still worth seeing, it is just not the headline.
        seen = [c for c in counts if c is not None]
        fell = any(b < a for a, b in zip(seen, seen[1:]))
        if len(set(shapes)) == 1 and not fell:
            continue
        row = f"{flag:<{width}}  " + "  ".join(
            f"{s + ('' if n is None else f'({n})'):>13}" for s, n in cells)
        (lost if fell else shape_only).append(row)

    for row in lost:
        print(row)
    for row in shape_only:
        print(row)

    if not lost and not shape_only:
        print("\nno lever changes shape or loses coverage across these rungs.")
        return 0
    print()
    if lost:
        print(f"{len(lost)} lever(s) SERVE FEWER call sites at a larger rung (listed first). "
              "This is the finding: root-cause each one, because the lever stopped "
              "covering work it used to cover and the wall pays for it.")
    if shape_only:
        print(f"{len(shape_only)} lever(s) change shape with their served count "
              "flat or rising. Usually benign -- a size-conditioned optimisation "
              "engaging, or an L1 ladder paying a refused probe before it serves.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
