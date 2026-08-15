#!/usr/bin/env python3
"""Grade this leg's arms with the e2e leg's grader, without touching the e2e leg's artifact.

e2e_disagree.py hard-codes its input and output directory to relion-scratch/e2e. Pointing its module
global at relion-scratch/fine and symlinking the reference in is the whole of this file: the grading
maths stays the e2e leg's, byte for byte, so the two legs' tables are comparable.

  python3 fine_disagree.py ref shape tt
"""
import json
import sys
from pathlib import Path

S = Path("/home/ttuser/relion-scratch")
FINE = S / "fine"
sys.path.insert(0, str(S / ".coworker-grade"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "relion-end-to-end"))
sys.path.insert(0, str(S))

import e2e_disagree as D  # noqa: E402

D.E2E = FINE

# The reference arm lives in e2e/. Link it in rather than copy: 1.9 MB per star and the grader only
# reads.
for name in ("ref_run_data.star",):
    dst = FINE / name
    if not dst.exists():
        dst.symlink_to(S / "e2e" / name)


def main():
    stems = sys.argv[1:] or ["ref_run", "shape_run", "tt_run"]
    base = stems[0]
    res = {"base": base, "pairs": {}}
    for s in stems[1:]:
        if not (FINE / f"{s}_data.star").exists():
            print(f"  {s}: no data.star yet, skipped", flush=True)
            continue
        res["pairs"][s] = D.grade(base, s)
    (FINE / "fine_disagree.json").write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {FINE / 'fine_disagree.json'}", flush=True)


if __name__ == "__main__":
    main()
