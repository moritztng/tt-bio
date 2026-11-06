"""Split one evaluated RFD4 arm into two disjoint halves, so quality.py can calibrate a floor from it.

    python3 perf/pvo_accuracy/rfd4_halves.py <evaluated arm root> <out root>

Writes `<out>/a` and `<out>/b`, each a `per_example_metrics.csv` of half the rows plus symlinks to that
half's design CIFs, which is the shape quality.py reads an arm in. Designs carry one seed each
(`rng="per_design"`), so the halves are disjoint seed sets as well as disjoint samples; the floor is still
recorded as `resample`, the weaker of the two claims.

Rows alternate by example_id order rather than splitting at the midpoint: generation order correlates with
the seed block, and a contiguous split would put a floor on one end of it.
"""
from __future__ import annotations

import csv
import pathlib
import sys


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    arm, out = (pathlib.Path(p).resolve() for p in argv)
    csvs = sorted(arm.rglob("per_example_metrics*.csv"))
    if len(csvs) != 1:
        raise SystemExit(f"expected exactly one per_example_metrics.csv under {arm}, found {len(csvs)}")
    with open(csvs[0], newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["example_id"])
    if len(rows) < 4:
        raise SystemExit(f"{len(rows)} rows is too few to halve")

    for name, half in (("a", rows[0::2]), ("b", rows[1::2])):
        d = out / name
        d.mkdir(parents=True, exist_ok=True)
        with open(d / "per_example_metrics.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(half)
        for row in half:
            src = pathlib.Path(row["designed_structure_path"])
            if not src.exists():
                continue
            link = d / src.name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(src)
        print(f"{d}: {len(half)} designs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
