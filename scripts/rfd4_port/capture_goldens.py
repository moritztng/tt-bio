"""Capture upstream feature goldens for every shipped RFD4 benchmark example.

One .npz per input under state/rfd4/goldens-io/<family>/<stem>.npz, plus an INDEX.json
naming the upstream sha, the input path and the arrays captured. Parity for this row is
scored on every family, not on one example, so the corpus is the whole benchmark tree.

Runs in the upstream reference env only. Resumable: an existing .npz is skipped.

    rfd4-ref-env/venv/bin/python -I capture_goldens.py [--jobs N] [--limit-per-family N]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

BENCH = Path("/home/moritz/rfd4-upstream/RFD4-Proteina/data/benchmarks")
OUT = Path("/home/moritz/.coworker/state/rfd4/goldens-io")
DUMP = Path(__file__).with_name("dump_upstream_features.py")
PY = "/home/moritz/rfd4-ref-env/venv/bin/python"


def inputs(limit_per_family=None):
    found = {}
    for p in sorted(BENCH.rglob("*.cif*")):
        if p.suffix not in (".zst", ".gz", ".cif"):
            continue
        family = p.relative_to(BENCH).parts[0]
        found.setdefault(family, []).append(p)
    if limit_per_family:
        found = {k: v[:limit_per_family] for k, v in found.items()}
    return found


def one(job):
    src, dst = Path(job[0]), Path(job[1])
    if dst.exists():
        return (str(src), "skip", 0.0, "")
    t0 = time.time()
    r = subprocess.run([PY, "-I", str(DUMP), str(src), str(dst)],
                       capture_output=True, text=True, timeout=1800)
    dt = time.time() - t0
    if r.returncode != 0:
        tail = (r.stderr.strip().splitlines() or [""])[-1][:300]
        return (str(src), "fail", dt, tail)
    return (str(src), "ok", dt, "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--limit-per-family", type=int, default=None)
    a = ap.parse_args()

    fams = inputs(a.limit_per_family)
    jobs = []
    for family, paths in fams.items():
        for p in paths:
            stem = p.name.split(".cif")[0]
            jobs.append((str(p), str(OUT / family / f"{stem}.npz")))
    print(f"{len(jobs)} inputs across {len(fams)} families: "
          + ", ".join(f"{k}={len(v)}" for k, v in sorted(fams.items())), flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for i, r in enumerate(ex.map(one, jobs), 1):
            results.append(r)
            if r[1] != "ok" or i % 10 == 0:
                print(f"[{i}/{len(jobs)}] {r[1]} {Path(r[0]).name} {r[2]:.1f}s {r[3]}", flush=True)

    ok = [r for r in results if r[1] == "ok"]
    skip = [r for r in results if r[1] == "skip"]
    fail = [r for r in results if r[1] == "fail"]
    index = {
        "upstream_sha": "2db07bb444d91695f8c10e6acb7648e9297f9221",
        "benchmarks": str(BENCH),
        "captured": len(ok) + len(skip),
        "failed": len(fail),
        "families": {k: len(v) for k, v in sorted(fams.items())},
        "failures": [{"input": r[0], "error": r[3]} for r in fail],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INDEX.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"captured {len(ok)} new, {len(skip)} already there, {len(fail)} failed", flush=True)


if __name__ == "__main__":
    main()
