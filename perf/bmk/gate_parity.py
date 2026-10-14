#!/usr/bin/env python3
"""One-line parity claim per page row, read from a release gate's Blackhole parity legs.

    python3 perf/bmk/gate_parity.py ~/gates/rel-<sha>

A gate leg PASSes when it reproduces its committed record, which is not the same claim as
"inside the reference envelope": a leg whose committed record is a documented bf16 floor
(GAP-evidenced) passes the gate while its own score reads GAP. So the claim is built from each
leg's own scorer verdict (the row its log prints under the leg name), and a
gate-level FAIL or ERROR is reported as that, never folded into a pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# page row id -> gate leg name prefixes that test it. Affinity legs are left off the Boltz-2 fold
# row (the fold cell runs no affinity head); Protenix v1 legs are not Protenix-v2.
LEGS = {
    "boltz2": ["boltz2-hsa", "boltz2-prot", "boltz2-trpcage", "boltz2-9ncy"],
    "esmfold2": ["esmfold2-trpcage", "esmfold2-cocrystal"],
    "esmfold2-fast": ["esmfold2-fast-"],
    "protenix-v2": ["protenix-9ncy", "protenix-hsa", "protenix-prot", "protenix-ubq"],
    "opendde": ["opendde-prot", "opendde-trpcage"],
    "openfold3": ["openfold3-"],
    "rf3": ["rf3-"],
    "openbind": ["openbind-"],
    "boltzgen": ["boltzgen"],
    "nesso1": ["nesso1"],
    "esmc-300m": ["esmc-300m"], "esmc-600m": ["esmc-600m"], "esmc-6b": ["esmc-6b"],
    "saprot-35m": ["saprot-35m"], "saprot-650m": ["saprot-650m"], "saprot-1.3b": ["saprot-1.3b"],
}
INSIDE = {"PASS", "PASS-caveated"}
VERDICTS = INSIDE | {"GAP", "GAP-evidenced", "DRIFT", "FAIL", "ERROR", "SKIP"}


def scorer_verdict(log: Path, leg: str) -> str | None:
    """The verdict on the leg's own result row; the header prints before the run, the row after."""
    rows = [l.split() for l in log.read_text(errors="replace").splitlines() if l.split()[:1] == [leg]]
    return next((w for w in rows[-1] if w in VERDICTS), None) if rows else None


def legs(gate: Path) -> dict:
    out = {}
    for line in (gate / "results.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["arch"] == "bh" and r["leg"].startswith("parity:"):
            out[r["leg"][7:]] = r   # newest result for a leg wins
    return out


def claims(gate: Path) -> dict:
    sha = gate.name.removeprefix("rel-")[:9]
    have = legs(gate)
    out = {}
    for row, prefixes in LEGS.items():
        mine = sorted(n for n in have if any(n.startswith(p) for p in prefixes)
                      and not (row == "esmfold2" and n.startswith("esmfold2-fast")))
        if not mine:
            continue
        inside, floor, failed = [], [], []
        for n in mine:
            r = have[n]
            log = Path(r.get("log") or "/nonexistent")
            v = scorer_verdict(log, n) if log.is_file() else None
            if r["verdict"] != "PASS" or v is None:
                failed.append(n)
            elif v in INSIDE:
                inside.append(n)
            else:
                floor.append(n)
        text = f"{len(inside)} of {len(mine)} release-gate accuracy legs inside the reference envelope on Blackhole, tt-bio {sha}"
        if floor:
            text += f"; {', '.join(floor)} outside it, reproducing a committed bf16 floor"
        if failed:
            text += f"; {', '.join(failed)} failed, errored or left no score in this gate"
        out[row] = text
    return out


def main() -> int:
    for row, text in claims(Path(sys.argv[1]).expanduser()).items():
        print(f"{row}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
