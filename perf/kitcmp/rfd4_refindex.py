#!/usr/bin/env python3
"""Index the B200 reference design set for rfd4-quality.

Reads the durable copy of a session's reference tree (generated designs, the
evaluation metrics the rescore phase wrote, and the per-run jsonl the generator
wrote) and emits one record per feature family: how many designs exist, which
seeds produced them, where the structures and the metric rows live locally, and
which metrics each row carries.

The paths inside the generated CSVs point at the rental box (/root/kc/...). The
index rewrites them to the local durable root so a grader never has to know a
destroyed instance's filesystem.
"""

import argparse
import csv
import json
import pathlib
import sys

PROV = {
    "example_id",
    "dataset_name",
    "task",
    "designed_structure_path",
    "output_base_dir",
    "metrics_tags",
    "chain_of_interest",
    "designed_chain_path",
}


def family(root: pathlib.Path, name: str) -> dict:
    gen = root / "reference" / name
    cells = sorted(p for p in gen.iterdir() if p.is_dir()) if gen.is_dir() else []
    seeds, arm, batch, size = set(), None, None, None
    structures = sorted(str(p.relative_to(root)) for p in gen.rglob("*_generated.cif"))
    for cell in cells:
        runs = cell / "runs.jsonl"
        if not runs.is_file():
            continue
        for line in runs.read_text().splitlines():
            r = json.loads(line)
            arm, batch, size = r.get("arm", arm), r.get("batch", batch), r.get("size", size)
            # runs.jsonl records the BASE seed on every line; rfd4_gpu.py runs batch i at
            # seed+i (`seed={a.seed + i}`), so the seed a design was made under comes from
            # its batch directory's index, not from the line.
            base = r["seed"]
            seeds |= {base + int(d.name[len("batch"):])
                      for d in cell.glob("batch*")
                      if d.is_dir() and any(d.rglob("*_generated.cif"))}

    rows, metrics = [], []
    csv_path = root / "reference-eval" / name / "evaluation_metrics" / "per_example_metrics.csv"
    if csv_path.is_file():
        with csv_path.open() as fh:
            rows = list(csv.DictReader(fh))
        metrics = [k for k in (rows[0] if rows else {}) if k not in PROV]

    return {
        "designs": len(structures),
        "metric_rows": len(rows),
        "seeds": sorted(seeds),
        "arm": arm,
        "size": size,
        "batch": batch,
        "metrics": metrics,
        "metrics_tags": sorted({t for r in rows for t in r["metrics_tags"].strip("[]").split(",") if t}),
        "structures_dir": str((gen).relative_to(root)),
        "metrics_csv": str(csv_path.relative_to(root)) if rows else None,
        "n_structures_listed": len(structures),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="durable reference tree (holds reference/ and reference-eval/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-designs", type=int, default=30)
    a = ap.parse_args()

    root = pathlib.Path(a.root).resolve()
    names = sorted(json.loads((root / "REFERENCE.json").read_text()))
    fams = {n: family(root, n) for n in names}

    short = {n: f["designs"] for n, f in fams.items() if f["designs"] < a.min_designs}
    nometric = [n for n, f in fams.items() if f["metric_rows"] != f["designs"]]
    index = {
        "root": str(root),
        "session": root.name,
        "gpu": "NVIDIA B200",
        "min_designs": a.min_designs,
        "families": fams,
        "short_families": short,
        "families_without_full_metrics": nometric,
        "ok": not short and not nometric,
    }
    pathlib.Path(a.out).write_text(json.dumps(index, indent=1) + "\n")

    for n, f in fams.items():
        print(f"{n:18s} {f['designs']:3d} designs  {f['metric_rows']:3d} metric rows  "
              f"seeds {f['seeds']}  arm={f['arm']} {f['size']}/b{f['batch']}  "
              f"{len(f['metrics'])} metrics")
    print(f"\nwrote {a.out}  ok={index['ok']}")
    return 0 if index["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
