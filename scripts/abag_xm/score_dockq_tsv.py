"""Score every sample structure of a campaign arm against the ground-truth references -> TSV.

Walks a pulled structures tree of the layout

    <structs>/<target>/<model>_results_<target>/structures/<target>.cif          (rank 0)
    <structs>/<target>/<model>_results_<target>/structures/<target>_model_<r>.cif (rank r)

and emits the ``target<TAB>rank<TAB>global_dockq`` TSV that
``build_scaling_dataset.py --dockq-tsv`` joins into the samples parquet. Exists as a script
rather than a shell loop because this campaign twice published numbers computed in ad-hoc
heredocs; the rank convention above (winner is rank 0, ``_model_<r>`` is rank r) is the
join key and must not be re-improvised per arm.

The scorer is ``scripts/opendde_dockq.py`` (reference DockQ 2.1.3), so run with an
interpreter that has DockQ installed. rc=2 from the scorer means no native interface had a
real DockQ; the row is written as ERR, which the parquet builder skips.

Usage:
    <dockq-env>/bin/python3 scripts/abag_xm/score_dockq_tsv.py \
        --structs /path/to/boltz2_structs --out /path/to/boltz2_dockq.tsv --jobs 16
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

RANKED = re.compile(r"^(?P<target>[A-Za-z0-9]+)(?:_model_(?P<rank>\d+))?\.cif$")


def one(job: tuple[str, str, str, int]) -> tuple[str, int, str]:
    model, native, target, rank = job
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve().parents[1]
                                             / "opendde_dockq.py"), model, native],
                       capture_output=True, text=True)
    if r.returncode == 2:
        return target, rank, "ERR"
    if r.returncode != 0:
        return target, rank, f"ERR(rc={r.returncode}: {r.stderr[-200:]})"
    try:
        d = json.loads(r.stdout)
    except json.JSONDecodeError:
        return target, rank, "ERR(unparseable scorer output)"
    return target, rank, f"{d['global_dockq']:.4f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--structs", required=True, type=pathlib.Path)
    ap.add_argument("--gt", type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parents[2]
                    / "examples" / "ground_truth_structures")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    jobs: list[tuple[str, str, str, int]] = []
    for sdir in sorted(args.structs.glob("*/*/structures")):
        target = sdir.parent.parent.name
        native = args.gt / f"{target}.cif"
        if not native.exists():
            continue  # only 34 of the 164 panel targets have a reference; skip the rest
        for cif in sorted(sdir.glob("*.cif")):
            m = RANKED.match(cif.name)
            if not m or m["target"] != target:
                print(f"  UNEXPECTED filename {cif} -- rank unknown, refusing to guess",
                      file=sys.stderr)
                return 1
            jobs.append((str(cif), str(native), target, int(m["rank"] or 0)))
    if not jobs:
        print("nothing to score", file=sys.stderr)
        return 1

    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        rows = list(ex.map(one, jobs))
    bad = [r for r in rows if r[2].startswith("ERR(")]
    for t, rank, msg in bad:
        print(f"  {t} rank {rank}: {msg}", file=sys.stderr)

    with args.out.open("w") as fp:
        for t, rank, v in sorted(rows, key=lambda r: (r[0], r[1])):
            fp.write(f"{t}\t{rank}\t{v}\n")
    scored = sum(1 for r in rows if not r[2].startswith("ERR"))
    print(f"{scored}/{len(rows)} scored across {len({r[0] for r in rows})} targets -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
