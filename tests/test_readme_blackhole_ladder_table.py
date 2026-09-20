"""The README's Blackhole ladder table has to say what the ladder actually recorded.

The paragraph this replaced was prose, and it went stale inside three days. It
told a user that `rf3` "does not return at 896" on a p300c and to treat anything
over 768 residues there as a known gap. By 2026-09-17 the recorded p300c ladder
had `rf3` at 896, 1024 and 1088, and by 2026-09-20 every other structure model
walked p150a to 1536. Nothing was checking, because a sentence is not a number.

So the table is generated from the same files the size-ladder recorder writes,
and this fails if the two drift. The ceiling table higher up the README is a
different invariant -- that one is held to `size_limits.CEILINGS`, the guard that
refuses a job, and only `opendde` has a Blackhole entry there. This one is a
record of what folded, which on Blackhole is most of what a user has.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
FRAGMENTS = ROOT / "docs" / "size_ladder_baseline.d"
CARDS = ("p150a", "p300c")
HEADER = "| model | p150a | p300c |"


def recorded_tops() -> dict[str, dict[str, int]]:
    """Largest rung each model has a runtime for, per Blackhole card."""
    tops: dict[str, dict[str, int]] = {}
    for path in sorted(FRAGMENTS.glob("*.json")):
        for card, entry in json.loads(path.read_text())["cards"].items():
            if card not in CARDS:
                continue
            for model, row in entry.get("models", {}).items():
                runtimes = row.get("runtime_s") or {}
                if runtimes:
                    tops.setdefault(model, {})[card] = max(int(k) for k in runtimes)
    return tops


def published_tops() -> dict[str, dict[str, int]]:
    lines = README.read_text().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip() == HEADER)
    out: dict[str, dict[str, int]] = {}
    for line in lines[start + 2:]:
        if not line.startswith("|"):
            break
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        model = re.findall(r"`([^`]+)`", cells[0])[0]
        out[model] = {
            card: int(cell) for card, cell in zip(CARDS, cells[1:]) if cell != "--"
        }
    return out


def test_the_table_matches_the_recorded_ladder():
    assert published_tops() == recorded_tops(), (
        "the README Blackhole ladder table and docs/size_ladder_baseline.d/ disagree. "
        "The table publishes the largest rung on record per card; re-derive it from the "
        "fragments rather than editing the number by hand."
    )


def test_the_table_reads_a_card_the_fragments_record():
    """Negative control: the reader has to actually find rows, not pass on an empty set.

    A regex that matched nothing would make the comparison above vacuously true.
    """
    published = published_tops()
    assert len(published) >= 8, published
    assert all(set(v) <= set(CARDS) and v for v in published.values()), published
