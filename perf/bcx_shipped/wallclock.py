#!/usr/bin/env python3
"""Per-trajectory wall seconds for any BindCraft 2 campaign, from BindCraft 2's own clock.

This row built `stamp_log.py` to prefix a UTC stamp onto a campaign's stdout because the CSV
tables looked clockless. They are not. `!_Trajectories.csv` column 41 is `Timing` and it holds
`worker=<id>;start=<epoch>;design=<seconds>;compiled=<0|1>`, written at `campaign.py:190`;
`!_Refolded.csv` carries `reprediction=<seconds>` per MPNN candidate, written at
`MPNN_stage.py:274`. `campaign_output.campaign_timing` reads both and is what this calls, so
the numbers are BindCraft 2's and not a re-derivation.

That makes per-trajectory wall time recoverable for EVERY arm this campaign has run, including
the ones launched with no stamper and the ones that finished before anybody was watching.
`stamp_log.py` still earns its place: it timestamps the STAGE verdicts inside a trajectory,
which no CSV records.

    PYTHONPATH=/home/ttuser/bcx_e2e/bc2 python3 wallclock.py <project> [<project> ...]

`compiled=1` marks the trajectory that paid JAX's tracing and compilation, so it is not
comparable with the ones after it. It is printed rather than averaged away.
"""
from __future__ import annotations

import csv
import json
import pathlib
import sys

from bindcraft.campaign_output import campaign_timing, parse_timing


def rows(project: pathlib.Path):
    table = project / "1_Trajectories" / "!_Trajectories.csv"
    if not table.is_file():
        return []
    with table.open() as handle:
        return list(csv.DictReader(handle))


def arm(project: pathlib.Path) -> dict:
    stamp = project / "arm_stamp.json"
    return json.loads(stamp.read_text()) if stamp.is_file() else {}


def pool(stamp: dict) -> str:
    """The two harnesses in this campaign spell the same fields differently."""
    models = stamp.get("resolved_design_models") or stamp.get("design_models") or []
    shipped = stamp.get("shipped_model_pool", stamp.get("multimer_pool"))
    return f"{len(models)}x{'multimer_v3' if shipped else 'monomer'}" if models else "?"


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__.strip())
        return 2
    print(f"{'arm':<26} {'host':<14} {'pool':<14} {'tr':>3} {'len':>4} {'design_s':>9} "
          f"{'refold_s':>9} {'c':>1}  terminal")
    for path in sys.argv[1:]:
        project = pathlib.Path(path)
        stamp, timing = arm(project), campaign_timing(str(project))
        label, host = project.name, stamp.get("host", "?")
        for row in rows(project):
            clock = parse_timing(row.get("Timing", ""))
            entry = timing["designs"].get(row["design"], {})
            terminal = row.get("terminated") or "completed"
            print(f"{label:<26} {host:<14} {pool(stamp):<14} {row['trajectory']:>3} "
                  f"{row['length']:>4} {float(clock.get('design', 0.0)):>9.1f} "
                  f"{entry.get('reprediction_seconds', 0.0):>9.1f} "
                  f"{clock.get('compiled', '?'):>1}  {terminal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
