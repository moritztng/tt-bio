"""The campaign's defect ledger is a ROTATING file, so reading it alone reads only the tail.

`scripts/rotate_state_docs.sh` globs `state/*/*.md`, which includes `state/of3t/DEFECTS.md`, and
it archives the middle verbatim to `state/archive/` every thirty minutes past 256 KB. It deletes
nothing. But every guard that counted defects read the live file only, and by pass 324 that file
held **35 of 174** distinct entries and **10 of 57** UNFIXED. The archives themselves get
re-rotated, so the chain is `archive-archive-of3t-DEFECTS.*`, `archive-of3t-DEFECTS.*`,
`of3t-DEFECTS.*` and then the live file, oldest content first.

What that cost, measured at pass 324: `UNFIXED_TRIAGE.json` was regenerated from the tail and
silently dropped four USER-FACING defects (D30, D32, D56, D58), and the orchestrator's `VERDICT:`
was "corrected" from the true 174/57 to the tail's 35/10 because the audit reported the tail's
count and the audit was believed. The only check that noticed was the USER-FACING closure plan,
which refused to publish because its own item list disagreed with the shrunken triage. A count is
not wrong here so much as scoped, and nothing recorded the scope.

So: every DEFECTS-reading guard reads the UNION through this module, and a rotation can no longer
close a defect by moving it.
"""
from __future__ import annotations

import pathlib

STATE = pathlib.Path("/home/moritz/.coworker/state")
LIVE = STATE / "of3t" / "DEFECTS.md"
ARCHIVE = STATE / "archive"


def archive_chain(stem: str = "of3t-DEFECTS") -> list[pathlib.Path]:
    """The archives holding `stem`'s rotated middles, OLDEST CONTENT FIRST.

    Each re-rotation prefixes another `archive-`, so the most-prefixed file holds the oldest
    content. Sorting by that count descending puts the chain back in authoring order, which is
    what `status_vocab.statuses_by_defect` needs: it takes the LATEST heading's status, so the
    live file must come last or a rotated-out UNFIXED would win over its own repair.
    """
    if not ARCHIVE.is_dir():
        return []
    return sorted((p for p in ARCHIVE.iterdir() if stem in p.name and p.suffix == ".md"),
                  key=lambda p: -p.name.count("archive-"))


def defects_text(live: pathlib.Path | None = None) -> str:
    """The whole ledger: every archived middle in authoring order, then the live tail."""
    live = live or LIVE
    parts = [p.read_text() for p in archive_chain()]
    if live.is_file():
        parts.append(live.read_text())
    return "\n".join(parts)


def scope() -> dict:
    """What the union was assembled from, for an artifact that quotes a count."""
    return {"live": str(LIVE), "archives": [p.name for p in archive_chain()],
             "why": "DEFECTS.md rotates; the live file is the tail, not the ledger"}


if __name__ == "__main__":
    import re
    t = defects_text()
    nums = {int(m) for m in re.findall(r"^### D(\d+)(?:\.| UPDATE\b)", t, re.M)}
    live_nums = {int(m) for m in re.findall(r"^### D(\d+)(?:\.| UPDATE\b)", LIVE.read_text(), re.M)}
    print(f"union {len(nums)} distinct defects, live tail {len(live_nums)}")
    print("archives, oldest first:", [p.name for p in archive_chain()])
