#!/usr/bin/env python3
"""Does TT_BIO_APB_CONCAT_HEADS cost OpenFold3 1.254 A, or does it re-rank the same samples?

The 2026-09-26 10:28Z interleaved control read the gate's confidence-selected sample at
1.658 A with the lever off and 2.912 A with it on, and the row recorded that as a 1.254 A
accuracy cost. But the two arms' FIVE per-sample RMSDs are the same multiset in a different
order:

    off  1.658  2.929  1.652  1.700  3.620
    on   2.912  1.657  1.640  1.691  3.876

so the alternative reading is that the sample set barely moves and the confidence head's
ordering of it flips. Those two readings call for opposite decisions, and the gate's summary
line cannot tell them apart because it prints one number per arm.

This settles it by keeping the CIFs and measuring structure against structure, and it takes
the seed floor in the same session so the lever's effect has a denominator that was measured
rather than borrowed from another model on another target.

Folds are interleaved off/on/off/on/off/off in one chain on one card. Every fold is its own
subprocess (one device context per process). No timing is recorded or claimed here: this is
an accuracy measurement and the host is loud.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WT = Path("/home/ttuser/.coworker/wt/land-standing")
OUT = WT / "perf/land_standing/out/apb_seedfloor"
PY = "/home/ttuser/tt-bio-dev/env/bin/python3"
CARD = os.environ.get("CARD", "3")
STEPS, SAMPLES = 200, 5

# off:s0 and on:s0 reproduce the 10:28Z control; s1 is its lever pair at a second seed;
# s2 and s3 are off-only and exist to make the seed floor a spread rather than one pair.
LEGS = [("off", 0), ("on", 0), ("off", 1), ("on", 1), ("off", 2), ("off", 3)]


def fold(arm, seed):
    tag = "%s_s%d" % (arm, seed)
    run = OUT / tag
    if (run / "openfold3_results_prot" / "structures").is_dir():
        print("[%s] already folded, reusing" % tag, flush=True)
        return run
    if run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True)
    env = dict(os.environ)
    env.update({
        "TT_VISIBLE_DEVICES": CARD,
        "TT_BIO_LEASE_CARDS": CARD,
        "TT_BIO_LEASE_HOLDER": "worker:land-standing",
        # The lever is driven from the environment, never a second default flip: wk/land-standing
        # is under a merge decision resting on it carrying exactly one non-comment line.
        "TT_BIO_APB_CONCAT_HEADS": "1" if arm == "on" else "0",
        "PYTHONPATH": str(WT),
    })
    cmd = [PY, "-m", "tt_bio.main", "predict", str(WT / "examples/prot.yaml"),
           "--model", "openfold3",
           "--sampling_steps", str(STEPS),
           "--diffusion_samples", str(SAMPLES),
           "--seed", str(seed),
           "--msa_dir", str(WT / "msa"),          # cached a3m, so the MSA is not a variable
           "--out_dir", str(run)]
    t0 = time.time()
    with open(run / "fold.log", "w") as log:
        rc = subprocess.call(cmd, cwd=str(WT), env=env, stdout=log, stderr=subprocess.STDOUT)
    print("[%s] rc=%d %.0fs" % (tag, rc, time.time() - t0), flush=True)
    if rc != 0:
        sys.exit("%s FAILED rc=%d, see %s" % (tag, rc, run / "fold.log"))
    return run


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {}
    for arm, seed in LEGS:
        runs["%s:s%d" % (arm, seed)] = str(fold(arm, seed))
    (OUT / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    print("FOLDS_DONE", flush=True)


if __name__ == "__main__":
    main()
