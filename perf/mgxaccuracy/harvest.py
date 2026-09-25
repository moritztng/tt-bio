#!/usr/bin/env python3
"""Flatten a fan record (`job.py --score-dsg`) into the row shape `twobytwo.py` reads.

The fan writes designability nested under `dsg` and says nothing about which ENGINE the job
ran on, because it does not know: the tree is chosen by whoever launched it. `twobytwo.py`
refuses a row with no `engine` key, on purpose -- this row's pre-fix and after-fix 1536
offset-0 cells are identical in every other field and differ by 2.1 A. So the engine label is
the one thing this script will not guess, and it is a required argument.

    python3 perf/mgxaccuracy/harvest.py --engine groel-onetree-ff5435cba \
        --in groel2x2.jsonl --out perf/mgxaccuracy/results/size_1536.jsonl

Every cell harvested by hand until now was transcribed by hand from the nested record, once
per cell, which is how a median ends up attached to the wrong offset. Nothing else changes:
the numbers come out of the fan's own `dsg` block untouched.
"""
import argparse
import json
import pathlib
import sys


def flatten(d, engine):
    """One fan record -> one twobytwo row, or None with a reason on stderr."""
    tag = d.get("tag", "?")
    if d.get("rc") != 0:
        return None, f"{tag}: rc={d.get('rc')}, not a measurement"
    dsg = d.get("dsg")
    if not dsg or not dsg.get("scrmsd"):
        return None, f"{tag}: no dsg.scrmsd -- was --score-dsg passed?"
    v = dsg["scrmsd"]
    if len(v) != dsg.get("n") or len(v) != d.get("n_designs"):
        return None, (f"{tag}: {len(v)} scrmsd values against n_designs="
                      f"{d.get('n_designs')}, dsg.n={dsg.get('n')} -- a partial cell is not "
                      f"a cell, score it from disk instead")
    row = {
        "model": d.get("model"),
        "target": d.get("target"),
        "crop_offset": d.get("crop_offset", 0),
        "target_res": d.get("target_res"),
        "side": "device",
        "engine": engine,
        "out_dir": d.get("out_dir") or f"~/mgxacc-work/out_*{tag}",
        "source": f"job.py --score-dsg, harvest.py from tag {tag!r}",
        "metric": dsg.get("column", "designfolding-bb_rmsd"),
        "backbone": "N,CA,C,O",
        "n": len(v),
        "median": dsg["median"],
        "min": dsg["min"],
        "max": dsg["max"],
        "pass_strict": dsg.get("pass_strict"),
        "pass_permissive": dsg.get("pass_permissive"),
        "wall_s": d.get("wall_s"),
        "aiclk_median": (d.get("aiclk") or {}).get("median"),
        "load_median": (d.get("load") or {}).get("median"),
        "card": d.get("card"),
        "ts": d.get("ts"),
        "scrmsd": v,
    }
    return row, None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="fan jsonl (local path)")
    ap.add_argument("--engine", required=True,
                    help="engine label for every row in this file. One tree per label: the "
                         "whole point of the field is that two cells on different trees "
                         "cannot be averaged by accident")
    ap.add_argument("--out", default="perf/mgxaccuracy/results/size_1536.jsonl")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows, refused = [], []
    for line in pathlib.Path(args.src).read_text().splitlines():
        if line.strip():
            row, why = flatten(json.loads(line), args.engine)
            (refused if row is None else rows).append(why or row)

    for why in refused:
        print(f"  refused {why}", file=sys.stderr)
    for r in rows:
        print(f"  {r['target_res']:>4} off {r['crop_offset']:>3}  n={r['n']}  "
              f"median {r['median']:.3f}  ({r['min']:.3f}-{r['max']:.3f})  "
              f"card {r['card']}  AICLK {r['aiclk_median']}  {r['wall_s']} s")
    if args.dry_run:
        print(f"\n  dry run, {len(rows)} row(s) NOT written to {args.out}")
        return 0
    with open(args.out, "a") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"\n  appended {len(rows)} row(s) to {args.out}")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())
