#!/usr/bin/env python3
"""Fold the same size-ladder fixture N times back to back inside ONE worker process.

The release gate spends one FRESH process per fold: release_gate._run_census_fold spawns
lever_census.py -> tt_bio.main predict for every rep at every rung, including the discarded
warm-up. So any cost paid once per worker process is paid on every gate fold at every rung,
and the per-rung warm-up discard cannot remove it. This harness measures that cost directly:
hand `predict` a directory of N identical copies and the scheduler folds them in one worker,
so fold 0 carries the per-process warm-up and folds 1..N-1 do not.

runtime_s is read from each target's own results.json, the exact field the gate records into
size_ladder_baseline.d. Order comes from the results.json mtimes, strictly increasing because
one worker folds sequentially.

    python3 fold_seq.py --rung 256 --folds 5 --tag q256p0 --out out/q256p0.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIXTURES = REPO / "perf" / "size512" / "fixtures"
STEPS = 6          # release_gate.SIZE_LADDER_STEPS
SEED = 0           # release_gate.SEED


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="boltz2")
    ap.add_argument("--rung", type=int, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    fixture = FIXTURES / ("cdk2x2_%d.yaml" % a.rung)
    if not fixture.exists():
        print("missing fixture %s" % fixture, file=sys.stderr)
        return 2

    work = HERE / "work" / a.tag
    shutil.rmtree(work, ignore_errors=True)
    indir = work / "in"
    indir.mkdir(parents=True)
    stems = ["%sf%d" % (a.tag, i) for i in range(a.folds)]
    for s in stems:
        shutil.copy(fixture, indir / (s + ".yaml"))
    out_dir = work / "out"

    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(indir),
           "--model", a.model, "--single_sequence",
           "--sampling_steps", str(STEPS), "--diffusion_samples", "1",
           "--seed", str(SEED), "--out_dir", str(out_dir)]
    log = work / "predict.log"
    t0 = time.monotonic()
    with open(log, "w") as fp:
        rc = subprocess.call(cmd, cwd=REPO, stdout=fp, stderr=subprocess.STDOUT)
    wall = time.monotonic() - t0

    rows = []
    for s in stems:
        hits = sorted(out_dir.glob("*_results_" + s))
        if not hits:
            continue
        rj = hits[0] / "results.json"
        if not rj.exists():
            continue
        try:
            recs = json.loads(rj.read_text())
        except Exception as e:
            print("unreadable %s: %s" % (rj, e), file=sys.stderr)
            continue
        ts = [r["runtime_s"] for r in recs
              if r.get("status") == "ok" and r.get("runtime_s") is not None]
        if ts:
            rows.append({"stem": s, "runtime_s": max(ts), "mtime": rj.stat().st_mtime})
    rows.sort(key=lambda r: r["mtime"])
    for i, r in enumerate(rows):
        r["order"] = i

    seq = [round(r["runtime_s"], 2) for r in rows]
    rec = {"tag": a.tag, "model": a.model, "rung": a.rung, "folds": a.folds,
           "rc": rc, "wall_s": round(wall, 2), "log": str(log),
           "host": os.uname().nodename,
           "tt_visible_devices": os.environ.get("TT_VISIBLE_DEVICES"),
           "loadavg": os.getloadavg(),
           "t_start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                        time.gmtime(time.time() - wall)),
           "t_end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "rows": rows,
           "sequence_s": seq}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rec, indent=2) + "\n")
    print("%s: rc=%d wall=%.1fs loadavg=%.2f sequence=%s"
          % (a.tag, rc, wall, os.getloadavg()[0], seq), flush=True)
    return 0 if rc == 0 and len(rows) == a.folds else 1


if __name__ == "__main__":
    raise SystemExit(main())
