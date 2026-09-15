#!/usr/bin/env python3
"""What the quiet re-capture has to show for each candidate to win, written before it is taken.

roof-true-floor's crossing is one number: the trunk pairformer's floor, 10.123 s, against its
measured 9.611 s at the cell, 105.3 %. The measured side is 13.708 s of a contended session scaled
by one scalar, 0.7011. The floor side is bytes and FLOPs from captures, which a re-capture at the
same head reproduces, plus per-class rates measured on pc.

So the re-capture decides it by a threshold, and the threshold is fixed now rather than after the
numbers land: the trunk's directly measured share of its own fold's wall.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRUNK = "PairformerLayer|1x512x384,1x512x512x128"


def main() -> int:
    tf = json.loads((HERE / "out_control_median" / "true_floor.json").read_text())
    bud = json.loads((HERE / "out_control_median" / "roof_budget_512_qb2c2.json").read_text())
    S = bud["summary"]
    row = next(r for r in tf["vs_measured"]["per_top_capture"] if r["capture"] == TRUNK)
    floor, meas = row["floor_s"], row["measured_at_cell_s"]
    cell, scale = S["cell_of_record_s"], S["cell_scale"]
    unscaled = meas / scale

    print("THE CROSSING, AS PUBLISHED")
    print("  trunk pairformer floor              %8.3f s" % floor)
    print("  trunk pairformer measured at cell   %8.3f s   %.1f %% of roof"
          % (meas, 100 * floor / meas))
    print("  that measured number is             %8.3f s of a contended session x %.4f"
          % (unscaled, scale))
    print()
    print("THE THRESHOLD THE QUIET RE-CAPTURE IS BEING TESTED AGAINST")
    print("  the trunk's share of the fold, as published   %6.2f %%  (%.3f / %.3f)"
          % (100 * meas / cell, meas, cell))
    print("  the share at which the crossing disappears    %6.2f %%  (%.3f / %.3f)"
          % (100 * floor / cell, floor, cell))
    print("  so the trunk has to come out %.2f points, %.1f %% relative, heavier than published."
          % (100 * (floor - meas) / cell, 100 * (floor - meas) / meas))
    print()
    print("  equivalently, the scale that would remove it   %.4f  against the published %.4f,"
          % (floor / unscaled, scale))
    print("  i.e. a session fold of %.3f s or less where the session measured %.3f s."
          % (cell / (floor / unscaled), S["session_fold_s"]))
    print()
    print("READ THE RESULT LIKE THIS")
    print("  trunk own-work >= %.2f %% of the quiet fold's own wall" % (100 * floor / cell))
    print("      -> CANDIDATE 1. The crossing was the 0.7011 rescale. Correct the floor with the")
    print("         directly measured times and report the new prize.")
    print("  trunk own-work <  %.2f %% of it" % (100 * floor / cell))
    print("      -> CANDIDATE 2. The rescale was not the cause, so at least one of roof-shape's")
    print("         per-class rates is not an upper bound for its class. Triangle attention is the")
    print("         named suspect: the fold reaches 20.6 % of the cube where the arms reach 14.7 %.")
    print()
    print("  Note the floor side moves too if the re-capture's bytes and FLOPs differ from the")
    print("  committed ones. At the same head they should not, and that is the control: the")
    print("  re-capture's 26 captures must reproduce %.4f TB and %.3f TFLOP."
          % (S["fold_TB"], S["fold_TFLOP_executed"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
