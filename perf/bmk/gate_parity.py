#!/usr/bin/env python3
"""One-line parity claim per page row, read from a release gate's Blackhole parity legs.

    python3 perf/bmk/gate_parity.py ~/gates/rel-<sha> [--extra state/site/msa-parity.json]

The page grades each model the way people run it: with an MSA wherever the model takes one. So a
`-nomsa` leg is a regression test for the gate and not part of the page's claim, and the Boltz-2
fold row reads its two MSA legs, which the gate carries as opt-in (they need the MSA server) and
`--extra` supplies from a separate run of the same harness.

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
    "boltz2": ["boltz2-prot-msa", "boltz2-ubiquitin-msa"],
    "esmfold2": ["esmfold2-trpcage", "esmfold2-cocrystal"],
    "esmfold2-fast": ["esmfold2-fast-"],
    "protenix-v2": ["protenix-9ncy", "protenix-hsa", "protenix-prot", "protenix-ubq"],
    "opendde": ["opendde-prot", "opendde-trpcage"],
    "openfold3": ["openfold3-"],
    "rf3": ["rf3-"],
    "openbind": ["openbind-"],
    "boltzgen": ["boltzgen"],
    "nesso1": ["nesso1"],
    "pxdesign": ["pxdesign-"],
    "esmc-300m": ["esmc-300m"], "esmc-600m": ["esmc-600m"], "esmc-6b": ["esmc-6b"],
    "saprot-35m": ["saprot-35m"], "saprot-650m": ["saprot-650m"], "saprot-1.3b": ["saprot-1.3b"],
}
# rows whose shipped default folds with an MSA; their claim says so
MSA_ROWS = {"boltz2", "openfold3", "protenix-v2", "opendde", "openbind", "rf3"}
VERDICTS = {"PASS", "PASS-caveated", "GAP", "GAP-evidenced", "DRIFT", "FAIL", "ERROR", "SKIP"}


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


def passed(gate: Path, extra: Path | None = None) -> dict:
    """{leg: True/False} for every Blackhole parity leg the page may count. A leg passes when the gate
    says PASS and its own scorer printed a verdict, so a leg that ran without scoring is not a pass."""
    out = {}
    for n, r in legs(gate).items():
        log = Path(r.get("log") or "/nonexistent")
        out[n] = r["verdict"] == "PASS" and log.is_file() and scorer_verdict(log, n) is not None
    if extra:
        for n, r in json.loads(extra.read_text()).get("legs", {}).items():
            out[n] = r.get("verdict") == "PASS" and bool(VERDICTS & set(str(r.get("scorer", "")).split()))
    return {n: v for n, v in out.items() if not n.endswith("-nomsa")}


def claims(gate: Path, extra: Path | None = None) -> dict:
    # A gate still running has only the legs it has reached so far, and every leg it has not
    # reached reads as missing. gate_fanout writes verdict.json last.
    if not (gate / "verdict.json").is_file():
        raise SystemExit(f"{gate}: no verdict.json, the gate has not finished; pass a finished gate")
    sha = gate.name.removeprefix("rel-")[:9]
    have = passed(gate, extra)
    side = json.loads(extra.read_text()) if extra else {}
    out = {}
    for row, prefixes in LEGS.items():
        mine = sorted(n for n in have if any(n.startswith(p) for p in prefixes)
                      and not (row == "esmfold2" and n.startswith("esmfold2-fast")))
        if not mine:
            continue
        ok = sum(have[n] for n in mine)
        what = "release-gate accuracy check" + ("s" if len(mine) > 1 else "")
        every = {1: "its", 2: "both"}.get(len(mine), f"all {len(mine)}")
        lead = f"Passes {every} {what}" if ok == len(mine) else f"{ok} of {len(mine)} {what} pass"
        at = side["sha"][:9] if any(n in side.get("legs", {}) for n in mine) else sha
        out[row] = (lead + " against the reference implementation on Blackhole" +
                    (", with an MSA" if row in MSA_ROWS else "") + f" (tt-bio {at})")
    return out


def main() -> int:
    extra = Path(sys.argv[3]).expanduser() if len(sys.argv) > 3 and sys.argv[2] == "--extra" else None
    for row, text in claims(Path(sys.argv[1]).expanduser(), extra).items():
        print(f"{row}: {text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
