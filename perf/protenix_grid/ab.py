#!/usr/bin/env python3
"""A/B one protenix-v2 fold config against another across the size ladder.

Answers two questions the issue-#9 grid workaround has to answer before it lands:
how much the 11x10 clamp costs at the size it fires (506 tokens), and whether it
would be cheaper to clamp everywhere (it is not, so the fix stays narrow). Arms are
interleaved rep by rep, so a box that gets busier partway through moves both arms.

Fold config is the size-ladder arm's: single-sequence, 6 steps, 1 sample, seed 0.

  python3 perf/protenix_grid/ab.py --rungs 506,512,768 --reps 2 --out perf/protenix_grid/ab.jsonl
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "perf" / "size512" / "fixtures"
ARMS = {
    "default": {},                              # whatever the engine picks per fold
    "g13": {"TT_BIO_FORCE_GRID": "13,10"},      # the grid that hangs in the window
    "g11": {"TT_BIO_FORCE_GRID": "11,10"},      # the grid that does not
}


def fold(rung, arm, workdir, timeout):
    out = workdir / f"{arm}_{rung}"
    shutil.rmtree(out, ignore_errors=True)
    env = dict(os.environ)
    env.pop("TT_BIO_FORCE_GRID", None)
    env.update(ARMS[arm])
    env["PYTHONPATH"] = str(ROOT)
    fixture = FIXTURES / f"cdk2x2_{rung}.yaml"
    cmd = [sys.executable, "-m", "tt_bio.main", "predict", str(fixture),
           "--model", "protenix-v2", "--single_sequence", "--sampling_steps", "6",
           "--diffusion_samples", "1", "--seed", "0", "--out_dir", str(out), "--debug"]
    log = workdir / f"{arm}_{rung}.log"
    t0 = time.monotonic()
    with open(log, "wb") as fh:
        rc = subprocess.call(cmd, cwd=ROOT, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             timeout=timeout)
    wall = time.monotonic() - t0
    rec = {"rung": rung, "arm": arm, "rc": rc, "wall_s": round(wall, 1)}
    text = log.read_text(errors="replace")
    for line in text.splitlines():
        if "grid" in line and "x10" in line and "compute" in line.lower():
            rec["grid_line"] = line.strip()[:160]
            break
    res = next(out.rglob("results.json"), None)
    if res:
        rows = json.loads(res.read_text())
        ok = [r for r in rows if r.get("status") == "ok"]
        if ok:
            rec["runtime_s"] = ok[0].get("runtime_s")
            rec["plddt"] = ok[0].get("plddt")
    cif = next(out.rglob("*.cif"), None)
    if cif:
        rec["cif_md5"] = hashlib.md5(cif.read_bytes()).hexdigest()
        rec["cif"] = cif.name
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="506,512,768")
    ap.add_argument("--arms", default="g13,g11,default")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default="perf/protenix_grid/ab.jsonl")
    ap.add_argument("--workdir", default="perf/protenix_grid/work")
    a = ap.parse_args()

    rungs = [int(r) for r in a.rungs.split(",")]
    arms = a.arms.split(",")
    workdir = ROOT / a.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    outp = ROOT / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)

    for rung in rungs:
        # One discarded fold per (rung, arm): the JIT kernel cache is keyed by shape AND
        # by program config, so neither another rung nor the other arm warms this one.
        for arm in arms:
            r = fold(rung, arm, workdir, a.timeout)
            r["rep"] = "warmup"
            print(json.dumps(r), flush=True)
            with open(outp, "a") as fh:
                fh.write(json.dumps(r) + "\n")
        for rep in range(a.reps):
            for arm in arms:
                r = fold(rung, arm, workdir, a.timeout)
                r["rep"] = rep
                print(json.dumps(r), flush=True)
                with open(outp, "a") as fh:
                    fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
