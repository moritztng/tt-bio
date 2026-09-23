#!/usr/bin/env python3
"""Does a model's host featurization depend on the process's string-hash seed?

hashseed.py <out_dir> <model>:<input.yaml> [...] [--seeds 0,1,2,3]

Runs `tt-bio predict` once per (job, PYTHONHASHSEED) with feats/sitecustomize.py on the path, so
the real parse/MSA/template/featurize path runs and the model is replaced by a stand-in that
hashes what it is handed and stops. No device is opened. Prints, per job, the number of distinct
feature sets across seeds and the keys that differ. A set or a str-keyed hash iterated into
feature order shows up here as a job with more than one distinct set.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("jobs", nargs="+")
ap.add_argument("--seeds", default="0,1,2,3")
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
bad = 0
for job in a.jobs:
    model, y = job.split(":", 1)
    tag = f"{model}_{Path(y).stem}"
    sets = {}
    for s in a.seeds.split(","):
        f = out / f"{tag}_h{s}.txt"
        f.unlink(missing_ok=True)
        env = dict(os.environ, PYTHONHASHSEED=s, DET_FEATS=str(f), TT_BIO_LEASE_DIR=str(out / "leases"),
                   TT_METAL_LOGGER_LEVEL="FATAL",
                   PYTHONPATH=f"{REPO}:{REPO}/perf/mgx_det/feats")
        log = out / f"{tag}_h{s}.log"
        with open(log, "w") as fh:
            subprocess.run([sys.executable, "-m", "tt_bio.main", "predict", str(REPO / y),
                            "--model", model, "--accelerator", "tenstorrent", "--host_threads", "1",
                            "--out_dir", str(out / f"{tag}_h{s}"), "--override"],
                           cwd=REPO, env=env, stdout=fh, stderr=subprocess.STDOUT)
        if not f.exists():
            tail = log.read_text(errors="replace").strip().splitlines()[-3:]
            sets[s] = {"<no capture>": " | ".join(tail)[:300]}
            continue
        kv = {}
        for tok in f.read_text().split()[1:]:
            k, _, v = tok.partition("=")
            kv[k] = v
        sets[s] = kv
    distinct = {tuple(sorted(v.items())) for v in sets.values()}
    keys = sorted({k for v in sets.values() for k in v})
    diff = [k for k in keys if len({v.get(k) for v in sets.values()}) > 1]
    n = len(next(iter(sets.values())))
    print(f"{job}: {len(distinct)} distinct feature set(s) over seeds {a.seeds}, {n} keys"
          + (f"; differ: {' '.join(diff[:12])}" if diff else ""), flush=True)
    if "<no capture>" in keys:
        print(f"   no capture: {next(iter(sets.values())).get('<no capture>')}", flush=True)
    bad += len(distinct) > 1
sys.exit(1 if bad else 0)
