#!/usr/bin/env python3
"""Splice size-ladder fragments from gate record legs into docs/size_ladder_baseline.d/.

    python3 scripts/splice_ladder_fragments.py ~/gates/record-<sha9>/recorded [...]

A record leg writes the whole baseline file of its model as it was at the recorded commit, with
one card type's entry re-recorded (<dir>/<card_type>/<model>.json). Copying that file over the
repo's would also put back every other card type's entry as it was then, undoing any re-record
that landed on main since. So only the fragment's own card type entry is taken; every other key
of the repo's file stays. Later directories win when two carry the same model and card type.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BASELINE_DIR = Path(__file__).resolve().parent.parent / "docs" / "size_ladder_baseline.d"


def splice(recorded_dirs, baseline_dir: Path = BASELINE_DIR) -> list:
    """Return the (model, card_type) pairs whose entry changed."""
    changed = []
    for d in map(Path, recorded_dirs):
        for frag in sorted(d.glob("*/*.json")):
            card, dst = frag.parent.name, baseline_dir / frag.name
            if not dst.is_file():
                raise SystemExit(f"{frag}: no {dst} to splice into")
            base, new = json.loads(dst.read_text()), json.loads(frag.read_text())
            if card not in new.get("cards", {}):
                raise SystemExit(f"{frag}: carries no '{card}' entry")
            if base["cards"].get(card) != new["cards"][card]:
                base["cards"][card] = new["cards"][card]
                dst.write_text(json.dumps(base, indent=2) + "\n")
                changed.append((frag.stem, card))
    return changed


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for model, card in splice(sys.argv[1:]):
        print(f"spliced {card} into {BASELINE_DIR / (model + '.json')}")
