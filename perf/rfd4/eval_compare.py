#!/usr/bin/env python3
"""Diff two `per_example_metrics.csv` files column for column.

This is the instrument behind every "exact vs upstream" claim the `rfd4-eval` row makes: one CSV
from upstream's `rfd4-evaluate` in the reference venv, one from tt-bio's port, the same designs on
both sides. Nothing here imports upstream or tt-bio, so it runs under any python3.

    python3 -I perf/rfd4/eval_compare.py UPSTREAM.csv PORT.csv [--atol 1e-9] [--json out.json]

Designs are joined on `designed_structure_path` (basename), not `example_id`: upstream appends an
8-hex content hash to the stem, so the two arms only agree on the path.

Exit code 0 when every compared column matches within tolerance on every design, 1 otherwise.
A column present on one side only is a mismatch, not a skip -- a silently missing metric is the
failure mode this tool exists to catch (upstream drops *all* inline columns for a design when any
one metric raises).
"""

import argparse
import csv
import json
import math
import os
import sys

# Columns that describe the run rather than the design. Never compared.
PROVENANCE = {
    "example_id",
    "dataset_name",
    "designed_structure_path",
    "designed_chain_path",
    "output_base_dir",
    "input_json_path",
    "metrics_tags",
}

#: Column suffixes upstream itself treats as metadata rather than a metric
#: (rfproteina `proteina.py:231-250`): every refold's own file paths, which name the out_dir.
PROVENANCE_SUFFIXES = ("refolded_structure_path", "confidence_json_path")


def is_provenance(column):
    return column in PROVENANCE or column.endswith(PROVENANCE_SUFFIXES)


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    by_key = {}
    for row in rows:
        key = os.path.basename(row.get("designed_structure_path", "") or row.get("example_id", ""))
        if not key:
            raise ValueError(f"{path}: a row has neither designed_structure_path nor example_id")
        if key in by_key:
            raise ValueError(f"{path}: duplicate design {key}")
        by_key[key] = row
    return by_key


def as_number(value):
    """Return (is_number, float). Empty cells are a value in their own right, not zero."""
    if value is None or value == "":
        return False, None
    try:
        return True, float(value)
    except ValueError:
        return False, None


def cell_equal(a, b, atol, rtol):
    a_num, a_val = as_number(a)
    b_num, b_val = as_number(b)
    if a_num != b_num:
        return False
    if not a_num:
        return (a or "") == (b or "")
    if math.isnan(a_val) and math.isnan(b_val):
        return True
    return abs(a_val - b_val) <= atol + rtol * abs(b_val)


def compare(upstream, port, atol, rtol, only_prefix=None):
    report = {"designs": {}, "missing_from_port": [], "extra_in_port": [], "n_designs": 0}
    shared = sorted(set(upstream) & set(port))
    report["n_designs"] = len(shared)
    report["designs_upstream_only"] = sorted(set(upstream) - set(port))
    report["designs_port_only"] = sorted(set(port) - set(upstream))

    for design in shared:
        up, pt = upstream[design], port[design]
        cols = [c for c in up if not is_provenance(c) and not c.startswith("generation.")]
        if only_prefix:
            cols = [c for c in cols if c.startswith(only_prefix)]
        diffs = {}
        for col in cols:
            if col not in pt:
                report["missing_from_port"].append(col)
                diffs[col] = {"upstream": up[col], "port": None, "reason": "column absent"}
                continue
            if not cell_equal(up[col], pt[col], atol, rtol):
                diffs[col] = {"upstream": up[col], "port": pt[col], "reason": "value differs"}
        for col in pt:
            if is_provenance(col) or col.startswith("generation.") or col in up:
                continue
            if only_prefix and not col.startswith(only_prefix):
                continue
            report["extra_in_port"].append(col)
            diffs[col] = {"upstream": None, "port": pt[col], "reason": "column not in upstream"}
        report["designs"][design] = {"n_compared": len(cols), "diffs": diffs}

    report["missing_from_port"] = sorted(set(report["missing_from_port"]))
    report["extra_in_port"] = sorted(set(report["extra_in_port"]))
    report["exact"] = (
        bool(shared)
        and not report["designs_upstream_only"]
        and not report["designs_port_only"]
        and all(not d["diffs"] for d in report["designs"].values())
    )
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("upstream")
    ap.add_argument("port")
    ap.add_argument("--atol", type=float, default=0.0, help="absolute tolerance, default 0 (exact)")
    ap.add_argument("--rtol", type=float, default=0.0, help="relative tolerance, default 0 (exact)")
    ap.add_argument("--prefix", default=None, help="compare only columns with this prefix")
    ap.add_argument("--json", default=None, help="write the full report here")
    args = ap.parse_args()

    report = compare(load(args.upstream), load(args.port), args.atol, args.rtol, args.prefix)

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)

    print(f"designs compared: {report['n_designs']}")
    for side in ("designs_upstream_only", "designs_port_only"):
        if report[side]:
            print(f"{side}: {', '.join(report[side])}")
    for design, info in sorted(report["designs"].items()):
        if not info["diffs"]:
            print(f"  {design}: {info['n_compared']} columns exact")
            continue
        print(f"  {design}: {len(info['diffs'])} of {info['n_compared']} columns differ")
        for col, d in sorted(info["diffs"].items()):
            print(f"    {col}: upstream={d['upstream']!r} port={d['port']!r} ({d['reason']})")
    print("EXACT" if report["exact"] else "NOT EXACT")
    return 0 if report["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())
