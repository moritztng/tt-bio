"""Per-arm verdict journal for the release gate, so a run that dies does not cost the arms
that already decided.

Why this exists, with the measurement. The gate is one process that folds thirteen arms in
sequence and prints a single verdict at the end. On 2026-09-18 that cost two full runs on
tt-quietbox2: the host hard-reset at 11:03:23Z and again at 19:05:50Z, the second time
47 minutes in with EIGHT arms already PASS and nothing on disk that said so. Both times the
previous boot's journal ends mid-line with no shutdown sequence, so there is no signal to
catch and no exit code to read -- the run simply stops existing. Boot -1 lasted 7h59m and
boot -2 lasted 14h39m, so a 3h36m gate on that box under campaign load finishes maybe two
runs in three, and every restart re-folds arms whose answer was already known.

An arm's verdict is a property of (tree, host, knobs, arm). Nothing about it depends on
which process printed it, so it can be written down when it happens and read back by the
next run. That is all this module does.

The key is deliberately strict. It refuses to resume across a different commit, a dirty
tree, a different host, a different card type, or a different --fast / --diffusion_trace,
because every one of those changes what an arm would score. A resumed green that composed
arms from two trees would be worse than no resume at all: it would be a green nobody could
reproduce, which is the failure mode this campaign has already hit twice by reading a
verdict without checking which commit produced it.

The journal is JSONL, one line per arm decision, flushed and fsynced as it is written --
the whole point is to survive a process that is about to be killed without warning.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Every field here changes what an arm would score, so a difference in any of them makes a
# past verdict inapplicable. `package` is the installed tt_bio the gate actually scores,
# which is not necessarily the checkout it was launched from (see release_gate.py's import
# comment); recording it means a resume across two different installs is visible instead of
# silent.
KEY_FIELDS = ("commit", "dirty", "host", "card_type", "fast", "diffusion_trace", "package")

VERDICTS = ("PASS", "FAIL", "BLOCKED")

_HEADLINE = re.compile(r"^GATE (PASS|FAIL|BLOCKED)\b")

# Arm -> the distinguishing substrings of that arm's headlines in release_gate.py, across all
# three verdicts. Used only by --ingest, to back-fill a journal from a log written before
# journalling existed. Ingest REFUSES on a GATE line that matches no arm or more than one, so
# rewording a headline breaks the back-fill loudly instead of dropping an arm silently, and
# tests/test_gate_arm_journal.py asserts every string here is still literally in that file.
INGEST_MARKERS = {
    "fold-models":    ("cleared parse + ground-truth floor + geometry",
                       "a model missed parse, the ground-truth floor or geometry",
                       "every fold leg never opened a device"),
    "rf3-1024aa":     ("rf3 at 997 aa cleared the crystal floor",
                       "rf3 at 997 aa missed the crystal floor",
                       "rf3 at 997 aa never opened a device"),
    "rfd3-fusion":    ("both fusion levers serve their expected rate",
                       "rfd3 fusion census"),
    "boltzgen":       ("boltzgen designs cleared parse + designability floor",
                       "boltzgen missed parse or the designability floor",
                       "boltzgen never opened a device"),
    "rfd3":           ("rfd3 designs cleared parse, designed-region geometry",
                       "rfd3 missed parse, geometry, sequence or determinism",
                       "rfd3 designs never opened a device"),
    "pxdesign":       ("pxdesign designs cleared parse + the conditioning floor",
                       "pxdesign missed parse or the conditioning floor",
                       "pxdesign never opened a device"),
    "opendde-abag":   ("opendde-abag cleared parse + DockQ floor",
                       "opendde-abag missed parse or the DockQ floor",
                       "opendde-abag never opened a device"),
    "nesso1":         ("nesso1 scalars cleared the reference floor",
                       "nesso1 missed the reference floor or drifted run to run",
                       "nesso1 never opened a device"),
    "capacity":       ("largest-input folds fit the DRAM budget",
                       "capacity regression at the largest supported input",
                       "capacity never opened a device"),
    "size-ladder":    ("lever census and scaling exponents",
                       "size-ladder drift vs the recorded baseline"),
    "l1-budget":      ("every part class can narrow out of a clash",
                       "a part class cannot escape an L1/CB clash"),
    "batch-position": ("identical targets are identical whatever their batch position",
                       "a result depends on a target's position in the batch"),
    "esmc":           ("ESMC embed path cleared the per-residue PCC floor",
                       "an ESMC model missed the per-residue PCC floor",
                       "the ESMC embed path never opened a device"),
}


def verdict_of(headline: str) -> str | None:
    """PASS / FAIL / BLOCKED off a gate headline, or None if the line is not one."""
    m = _HEADLINE.match(headline.strip())
    return m.group(1) if m else None


def append(path: Path, key: dict, arm: str, verdict: str, headline: str,
           source: str | None = None, members=None) -> None:
    """Append one arm decision. Best effort by design: a journal write must never be able to
    fail a gate leg that has already produced a real answer.

    `members` is the set of --model values this arm's single verdict covered. Two legs score
    several models under one headline, so without it a narrow run would discharge a wide one.
    """
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}, got {verdict!r}")
    rec = {"ts": time.time(), "arm": arm, "verdict": verdict,
           "headline": headline.strip().splitlines()[0], "key": {k: key.get(k) for k in KEY_FIELDS},
           "members": sorted(members) if members else [arm]}
    if source:
        rec["source"] = source
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    except OSError as exc:                                  # pragma: no cover - disk failure
        print(f"[gate-journal] could not record {arm}: {exc}", file=sys.stderr)


def read(path: Path) -> list[dict]:
    """Every record, oldest first. A truncated last line is dropped rather than raising: the
    process that wrote it was killed, which is the case this module exists for."""
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def same_key(a: dict, b: dict) -> bool:
    return all(a.get(f) == b.get(f) for f in KEY_FIELDS)


def resumable(path: Path, key: dict) -> dict[str, dict]:
    """arm -> its record, for arms whose LATEST decision under an identical key is a PASS.

    Latest, not any: an arm that passed and was then re-run to a FAIL on the same tree is a
    failing arm. A dirty tree is never resumable, because it has no stable identity -- two
    runs at the same commit with different uncommitted edits would key identically.
    """
    if key.get("dirty"):
        return {}
    latest: dict[str, dict] = {}
    for rec in read(path):
        if same_key(rec.get("key") or {}, key):
            latest[rec["arm"]] = rec
    return {arm: rec for arm, rec in latest.items() if rec.get("verdict") == "PASS"}


def ingest(log: Path, key: dict, journal: Path) -> list[tuple[str, str]]:
    """Back-fill a journal from a gate log written before journalling existed.

    Returns the (arm, verdict) pairs recorded. Raises on any GATE headline it maps to zero or several
    arms, so this cannot quietly under-report what a log contains -- an unrecognised
    headline means the table below is stale, and a resume built on a stale table would skip
    an arm that never ran.
    """
    recorded = []
    for n, raw in enumerate(log.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        verdict = verdict_of(raw)
        if verdict is None:
            continue
        hits = [a for a, markers in INGEST_MARKERS.items() if any(m in raw for m in markers)]
        if len(hits) != 1:
            raise ValueError(
                f"{log}:{n}: this gate headline maps to {hits or 'no arm'}, so the ingest "
                f"table in gate_journal.INGEST_MARKERS is stale:\n  {raw}")
        arm = hits[0]
        append(journal, key, arm, verdict, raw, source=f"{log}:{n}")
        recorded.append((arm, verdict))
    return recorded


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("journal", type=Path, help="Path to the JSONL journal.")
    ap.add_argument("--ingest", type=Path, default=None, metavar="GATE.LOG",
                    help="Back-fill arm verdicts from an existing gate log. Needs the key "
                         "flags below to say which tree and host produced it.")
    ap.add_argument("--show", action="store_true", help="Print the journal, one arm per line.")
    for f in KEY_FIELDS:
        ap.add_argument(f"--{f.replace('_', '-')}", default=None)
    args = ap.parse_args(argv)

    if args.ingest:
        key = {f: getattr(args, f) for f in KEY_FIELDS}
        key["dirty"] = str(key["dirty"]).lower() in ("1", "true", "yes")
        for f in ("fast", "diffusion_trace"):
            key[f] = str(key[f]).lower() in ("1", "true", "yes")
        for arm, verdict in ingest(args.ingest, key, args.journal):
            print(f"{arm:<16}{verdict}")
        return 0

    if args.show:
        for rec in read(args.journal):
            k = rec.get("key") or {}
            print(f"{time.strftime('%H:%M:%SZ', time.gmtime(rec['ts']))}  "
                  f"{rec['arm']:<16}{rec['verdict']:<9}{k.get('commit')}  {k.get('host')}")
        return 0

    ap.error("give --ingest or --show")
    return 2


if __name__ == "__main__":
    sys.exit(_main())
