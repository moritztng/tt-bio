#!/usr/bin/env python3
"""Re-score reference designs that were generated without metrics, in place, on the box that made them.

Generation writes `inference_metrics/per_example_metrics.csv` beside the CIFs, but that file carries only
config and provenance columns -- the generate path does not run the metric suite, so every row that later
reads it for a metric finds none (confirmed on the GPU session 4 tree, 2026-10-21, same defect rfd4-quality
root-caused on CPU). `rfd4-evaluate inputs=<dir> metrics_tags='[...]' out_dir=<dir>` scores the CIFs that
already exist, so this costs no generation time and must happen before the rental ends.

Each family is scored with the tags REFERENCE.json recorded for it, so the upstream reference set and the TT
set are gated by the same metrics. A family whose tags need an oracle the box does not have falls back to
`[design]`, the oracle-free inline set, and the fallback is recorded rather than hidden.

    rfd4_rescore.py <deadline-epoch> [out_root]
"""
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

B = Path("/root/kc/rfd4")
EVAL = "/opt/rfd4-venv/bin/rfd4-evaluate"
FALLBACK = ["design"]


def columns(p):
    try:
        with open(p, newline="") as f:
            r = csv.reader(f)
            head = next(r, [])
            return head, sum(1 for _ in r)
    except OSError:
        return [], 0


def provenance_columns():
    """The column set the generate path already writes, so "has a metric" means "has a column beyond these"."""
    seen = set()
    for p in B.glob("reference/*/*/batch*/inference_metrics/per_example_metrics.csv"):
        head, _ = columns(p)
        seen |= set(head)
    return seen


def scored(out_dir, prov):
    """(rows, metric columns) of the evaluate tree under out_dir, merged over however many CSVs it wrote."""
    rows, mets = 0, set()
    for p in out_dir.rglob("per_example_metrics.csv"):
        head, n = columns(p)
        rows += n
        mets |= {c for c in head if c not in prov}
    return rows, sorted(mets)


def evaluate(inputs, out_dir, tags, deadline):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [EVAL, f"inputs={inputs}", "metrics_tags='[%s]'" % ",".join(tags), f"out_dir={out_dir}"]
    print(f"{time.strftime('%FT%TZ', time.gmtime())} -> {' '.join(cmd)}", flush=True)
    t = time.time()
    try:
        r = subprocess.run(cmd, timeout=max(60, deadline - int(time.time())), check=False,
                           capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        print("  TIMEOUT", flush=True)
        return None
    if r.returncode:
        print(f"  rc={r.returncode} in {time.time() - t:.0f}s: {(r.stderr or '')[-600:]}", flush=True)
    return r.returncode


def main():
    deadline = int(sys.argv[1])
    root = Path(sys.argv[2]) if len(sys.argv) > 2 else B / "reference-eval"
    prov = provenance_columns()
    ref = json.loads((B / "REFERENCE.json").read_text())
    done = {}
    for fam, cfg in ref.items():
        if time.time() > deadline - 120:
            print(f"deadline: {fam} and everything after it was not rescored", flush=True)
            break
        fam_dir = B / "reference" / fam
        if not fam_dir.is_dir():
            continue
        for tags in (cfg.get("metrics_tags") or FALLBACK, FALLBACK):
            out_dir = root / fam
            evaluate(fam_dir, out_dir, tags, deadline)
            rows, mets = scored(out_dir, prov)
            if rows and mets:
                break
            print(f"rescore {fam}: nothing scored with tags {tags}", flush=True)
            if tags == FALLBACK:
                break
        done[fam] = {"rows": rows, "metrics": mets, "metrics_tags": tags,
                     "fallback": tags == FALLBACK and tags != (cfg.get("metrics_tags") or FALLBACK),
                     "out": str(out_dir), "designs": cfg.get("designs")}
        print(f"rescore {fam}: {rows} rows, {len(mets)} metric columns, tags {tags}", flush=True)
        (B / "RESCORE.json").write_text(json.dumps(done, indent=2) + "\n")
    (B / "RESCORE.json").write_text(json.dumps(done, indent=2) + "\n")
    print(json.dumps(done, indent=2))


if __name__ == "__main__":
    main()
