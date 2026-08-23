#!/usr/bin/env python3
"""Is the AF2-IG device trunk still on its known bfloat16 floor, or is it something new?

This leg cannot be registered as a PASS leg. The committed device report misses 9 of 94 taps
and 3 of 6 scalars, and passes 9-11 established what that residual is: a bfloat16 realisation
floor in the residual chain, amplified about 3x by a structure module doing its job correctly,
with trunk-precision work provably dead against it (state doc, VERDICT-P11). The gate already
has the mechanism for exactly this shape -- a live GAP that reproduces a committed
``GAP-evidenced`` record is gate-passing, while a FAIL is excused by nothing
(``full_parity_gate.py:_matches_committed``).

So the leg's entire content is this classifier: GAP when the residual reproduces the committed
floor, FAIL when it does not. It reads the report's ``rows`` and ``scalars`` and never the
report's ``verdict`` field, because ``tap_gate.py --mutate`` inverts that field -- a classifier
keyed on it could not be controlled at all.

Four conditions, every one of them against the committed record rather than a list written out
here, so there is a single source of truth for what the floor is:

  * the same taps are scored and none went missing (a vanished tap is a regression, pass 6);
  * every failing tap is one the committed record already fails;
  * every failing measure's ``envelope_ratio`` is within ``tol`` of its committed value;
  * every failing scalar is one the committed record already fails, and its ``delta`` is within
    ``tol`` of the committed delta.

The two ratio bounds are what make the name allowlist honest. A mutation that only worsens the
9 taps which already fail is invisible to a check on names alone, and ``--mutate extra-const``
is exactly that shape: it zeroes the constant the dead extra-MSA track collapses to, which
moves the trunk without adding a new failing tap.

    PYTHONPATH=. python3 scripts/af2_port/device_floor.py --report live.json \
        --committed docs/implementation-parity-data/af2ig-trunk-device.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

#: How far a failing measure may drift from its committed value and still count as the same
#: floor. Measured, not chosen: the device arm has no rerun noise inside one process (pass 12,
#: 18/18 comparisons at 0.0), but this leg has run on cards 0, 2 and 3 across passes and ttnn is
#: not promised to be bit-exact across cards, so the tolerance is 1.25x the cross-card spread
#: measured on qb1 cards 3 and 0, floored at 1.10 because a tolerance tighter than the spread is
#: a flake generator. Pass 13 measured a spread of 1.0 (both cards bit-identical on all 94 taps
#: and all 6 scalars), so the floor is what binds.
DEFAULT_TOL = 1.10

_PASSING_ROW = ("PASS", "IN-ENVELOPE")


def _failing_taps(report: dict) -> dict:
    """{tap: envelope_ratio} for every scored tap that missed its bars.

    ``rows`` rather than ``failures``, which tap_gate truncates to 12.
    """
    return {r["tap"]: r.get("envelope_ratio")
            for r in report.get("rows", []) if r.get("verdict") not in _PASSING_ROW}


def _failing_scalars(report: dict) -> dict:
    return {r["scalar"]: r.get("delta")
            for r in report.get("scalars", []) if r.get("verdict") not in _PASSING_ROW}


def af2ig_device_floor_verdict(report: dict, committed: dict,
                               tol: float = DEFAULT_TOL) -> tuple[str, str]:
    """(verdict, detail) for one live device-trunk report against the committed floor.

    GAP iff the live residual is the committed one; FAIL for anything else, including a report
    that never reached the envelope arm and so carries no ratio to bound.
    """
    # The absent-prerequisite GAP is set upstream by the gate and carries its reason in
    # `error`, so it is read before the error branch rather than after it.
    if isinstance(report, dict) and report.get("verdict") == "GAP":
        return "GAP", str(report.get("error", "prerequisite absent"))
    if not isinstance(report, dict) or report.get("error"):
        return "ERROR", str((report or {}).get("error", "no report"))
    live_rows = report.get("rows") or []
    if not live_rows:
        return "NO-DATA", "no taps scored"

    reasons = []
    # 1. structural: same taps scored, none vanished.
    if report.get("not_implemented"):
        reasons.append("taps not produced: %s" % ", ".join(report["not_implemented"][:3]))
    want_scored = committed.get("taps_scored")
    if want_scored is not None and report.get("taps_scored") != want_scored:
        reasons.append("scored %s taps, committed %s" % (report.get("taps_scored"), want_scored))

    # 2 + 3. the failing taps and how far they miss.
    live, floor = _failing_taps(report), _failing_taps(committed)
    for tap in sorted(set(live) - set(floor)):
        reasons.append("new failing tap %s" % tap)
    for tap in sorted(set(live) & set(floor)):
        got, want = live[tap], floor[tap]
        if got is None or want is None:
            reasons.append("%s has no envelope ratio to bound" % tap)
        elif got > tol * want:
            reasons.append("%s ratio %.4f > %.2fx committed %.4f" % (tap, got, tol, want))

    # 4. the scalars, on the raw delta rather than a ratio: the scalar envelope is itself a
    # measured width and both arms of it move with the port.
    live_s, floor_s = _failing_scalars(report), _failing_scalars(committed)
    for name in sorted(set(live_s) - set(floor_s)):
        reasons.append("new failing scalar %s" % name)
    for name in sorted(set(live_s) & set(floor_s)):
        got, want = live_s[name], floor_s[name]
        if got is None or want is None:
            reasons.append("scalar %s has no delta to bound" % name)
        elif got > tol * want:
            reasons.append("scalar %s delta %.6f > %.2fx committed %.6f"
                           % (name, got, tol, want))

    head = "%d/%d taps and %d/%d scalars on the committed bf16 floor (tol %.2fx)" % (
        len(live), report.get("taps_scored", len(live_rows)),
        len(live_s), len(report.get("scalars") or []), tol)
    if reasons:
        return "FAIL", "off the committed floor: " + "; ".join(reasons[:4])
    return "GAP", head


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="live tap_gate --device report")
    ap.add_argument("--committed", required=True, help="the GAP-evidenced record it must match")
    ap.add_argument("--tol", type=float, default=DEFAULT_TOL)
    args = ap.parse_args()
    verdict, detail = af2ig_device_floor_verdict(json.loads(Path(args.report).read_text()),
                                                 json.loads(Path(args.committed).read_text()),
                                                 tol=args.tol)
    print(json.dumps({"verdict": verdict, "detail": detail}, indent=1))
    return 0 if verdict == "GAP" else 1


if __name__ == "__main__":
    raise SystemExit(main())
