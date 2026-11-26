#!/usr/bin/env python3
"""Grade a loss-parity or loss-curve artifact against the pre-registered G3 bars.

The bars live in ``tt_bio/train/parity.py`` and were committed before any RFD4 loss or curve
existed. This script only applies them, so the verdict is mechanical: it prints the exact
``LOSS-PARITY:`` / ``CURVE:`` line that ``state/r4t/rows/accuracy.md`` carries, and exits
non-zero on FAIL.

Loss parity (``--parity FILE``). JSON::

    {"dtype": "bfloat16",
     "ours":  [{"total": .., "bb_ca": .., ...}, ...],     # >= 8 batches
     "ref":   [{"total": .., "bb_ca": .., ...}, ...],     # upstream, same batches
     "provenance": {...}}                                  # free-form, printed back

Both sides must use upstream's term names verbatim: a term one side has and the other lacks is
a FAIL, because a loss that silently drops a term matches on everything it still computes.

Curve (``--curve FILE``). JSON::

    {"ours": [per-step loss, ...], "gpu_a": [...], "gpu_b": [...], "warmup": 1000}

``gpu_b`` is upstream re-run at a second seed on the same GPU and is not optional: the bar is
``max(2 % of upstream's final-200 mean, 1.5 x the seed-to-seed gap)``, so without it a
cross-arch gap has no scale to be judged against.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tt_bio.train.parity import (CURVE_MIN_STEPS, LOSS_PARITY_BARS,  # noqa: E402
                                 LOSS_PARITY_MIN_BATCHES, curve_match, loss_parity)


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def do_parity(doc: dict) -> int:
    dtype = doc.get("dtype", "bfloat16")
    if dtype not in LOSS_PARITY_BARS:
        print(f"unknown dtype {dtype!r}; the bars are {sorted(LOSS_PARITY_BARS)}")
        return 2
    report = loss_parity(doc["ours"], doc["ref"], dtype)
    print(report)
    if prov := doc.get("provenance"):
        print("  provenance: " + json.dumps(prov, sort_keys=True))
    if len(doc["ref"]) < LOSS_PARITY_MIN_BATCHES:
        print(f"  note: the bar needs n >= {LOSS_PARITY_MIN_BATCHES} batches")
    return 0 if report.passed else 1


def do_curve(doc: dict) -> int:
    report = curve_match(doc["ours"], doc["gpu_a"], doc["gpu_b"], int(doc["warmup"]))
    print(report)
    if prov := doc.get("provenance"):
        print("  provenance: " + json.dumps(prov, sort_keys=True))
    if len(doc["gpu_a"]) < CURVE_MIN_STEPS:
        print(f"  note: the bar needs >= {CURVE_MIN_STEPS} steps")
    return 0 if report.passed else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--parity", metavar="FILE")
    g.add_argument("--curve", metavar="FILE")
    a = ap.parse_args()
    return do_parity(_load(a.parity)) if a.parity else do_curve(_load(a.curve))


if __name__ == "__main__":
    sys.exit(main())
