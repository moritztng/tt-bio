"""Run a condition family's designs through the CLI, with the clock sampled during the run.

    TT_VISIBLE_DEVICES=<card> python perf/rfd4/cond/run_designs.py --family <id> \
        --out ~/rfd4-cond-runs [--designs 2] [--seed 0] [--parent <cif>]

The design itself goes through `tt-bio design --model rfd4`, the path a user has, rather than
through the engine directly: the families this row owns are a condition spec and a featurizer on
one engine, and the spec only reaches the model through the CLI.

The recipe is the family's own (`families.py`), full length, every step and every recycle. AICLK
is sampled on a thread for the life of the run, because on Blackhole the clock sets the time;
under ~1200 MHz the line is flagged `clock_artifact` and the number is not a measurement of the
model.

Two designs in one invocation is the seed floor: one weight load, two independent per-design
noise streams, which is how far apart two honest designs of the same input land.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))        # perf/ for clocksample
sys.path.insert(0, str(HERE))                   # this directory for families

import clocksample  # noqa: E402
import families  # noqa: E402

CLOCK_FLOOR_MHZ = 1200


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(families.FAMILIES))
    ap.add_argument("--out", default=str(Path.home() / "rfd4-cond-runs"))
    ap.add_argument("--designs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--parent", default=None, help="partial diffusion: the design to restart from")
    a = ap.parse_args()

    out = Path(a.out).expanduser() / a.family
    out.mkdir(parents=True, exist_ok=True)
    cmd = families.command(a.family, out, seed=a.seed, num_designs=a.designs,
                           parent_design=Path(a.parent) if a.parent else None)
    # The same CLI, reached through this interpreter, so the run cannot pick up a tt-bio from a
    # different environment than the one it was launched in.
    cmd = [sys.executable, "-m", "tt_bio.main"] + cmd[1:]

    record = {
        "family": a.family,
        "command": " ".join(cmd),
        "recipe": families.FAMILIES[a.family]["recipe"],
        "seed": a.seed,
        "designs": a.designs,
        "card": os.environ.get("TT_VISIBLE_DEVICES"),
        "host": os.uname().nodename,
        "sha": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=HERE).stdout.strip(),
        "t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with clocksample.during(period=2.0) as clk:
        start = time.time()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - start
    summary = clk.summary()
    mhz = _median_mhz(summary)

    record.update(
        rc=proc.returncode,
        wall_s=round(wall, 2),
        s_per_design=round(wall / max(a.designs, 1), 2),
        aiclk=summary,
        aiclk_mhz=mhz,
        clock_line=clk.line(0),
        clock_artifact=bool(mhz is not None and mhz < CLOCK_FLOOR_MHZ),
        cifs=sorted(str(p) for p in out.glob("*_generated.cif")),
        stdout=proc.stdout[-4000:],
        stderr=proc.stderr[-4000:],
    )
    (out / "run.json").write_text(json.dumps(record, indent=2))
    with open(Path(a.out).expanduser() / "runs.jsonl", "a") as f:
        f.write(json.dumps({k: v for k, v in record.items()
                            if k not in ("stdout", "stderr")}) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k not in ("stdout", "stderr")}, indent=2))
    if proc.returncode:
        print(proc.stdout[-2000:], proc.stderr[-2000:], file=sys.stderr)
    return proc.returncode


def _median_mhz(summary):
    """The granted card's median AICLK. With TT_VISIBLE_DEVICES set it is device 0."""
    entry = summary.get(0) or (next(iter(summary.values())) if summary else None)
    return entry["median"] if entry else None


if __name__ == "__main__":
    raise SystemExit(main())
