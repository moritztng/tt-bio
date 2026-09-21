#!/usr/bin/env python3
"""Run another row's device instrument under the path census, unchanged.

    census.py --out CENSUS.json -- perf/of3t_diffusion/device_gradient.py --structs all ...

The instrument is executed by `runpy`, exactly as `perf/of3t_wholemodel/armrun.py` runs it, so
the arm being censused is the arm that produced the gradient artifact rather than a copy of it.
`sitecov.install()` is called first because the proxy caches a resolved verb into its instance
dict on first use, and a verb resolved before the patch would keep the unrecorded wrapper.

The report is written even when the instrument raises: a census of a run that died halfway is
still a census, and it says so in `instrument_rc`.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import runpy
import sys
import time

REPO = os.getcwd()
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "perf", "of3t_pathcov"))
sys.path.insert(0, os.path.join(REPO, "perf", "of3t_tape"))
sys.path.insert(0, os.path.join(REPO, "perf"))

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--no-lines", action="store_true",
                help="skip line coverage; the taped counts and the reach are unaffected")
ap.add_argument("rest", nargs=argparse.REMAINDER)
a = ap.parse_args()
argv = [x for x in a.rest if x != "--"]
if not argv:
    raise SystemExit("census.py: nothing to run")

import sitecov                                                        # noqa: E402
import tt_bio.autograd  # noqa: F401,E402  imported so the patch lands on a live module
import tt_bio.taped_ttnn  # noqa: F401,E402

n_sites = sitecov.install(REPO, want_lines=not a.no_lines)
print(f"pathcov: {n_sites} static ttnn call sites armed, running {argv[0]}", flush=True)

t0 = time.time()
sys.argv = list(argv)
rc, err = 0, None
try:
    runpy.run_path(argv[0], run_name="__main__")
except SystemExit as e:
    rc = e.code or 0
except BaseException as e:                                     # a census of a dead run is data
    rc, err = 1, f"{type(e).__name__}: {e}"
    import traceback
    traceback.print_exc()
elapsed = time.time() - t0

sitecov.release()
rep = sitecov.report()
rep["instrument"] = argv
rep["instrument_rc"] = rc
rep["instrument_error"] = err
rep["elapsed_s"] = round(elapsed, 1)
rep["host"] = os.uname().nodename
rep["card"] = os.environ.get("TT_VISIBLE_DEVICES")
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
# gzip by default: a census is 1.4 MB of per-site rows and four of them are 5.9 MB of git
# history for a file nothing reads by hand.
opener = gzip.open if a.out.endswith(".gz") else open
with opener(a.out, "wt") as f:
    json.dump(rep, f, indent=1)

ex = sum(1 for s in rep["sites"] if s["executed_in_tape"])
tp = sum(1 for s in rep["sites"] if s["taped_calls"])
hole = [s for s in rep["sites"] if s["raw_calls_in_tape"]
        and s["taped_verb"]]
print(f"pathcov: {rep['n_tape_opens']} tape opens, {ex} sites executed in a tape, "
      f"{tp} taped, {len(hole)} raw-ttnn-in-tape, {rep['n_leaves']} parameter leaves, "
      f"{len(rep['untracked_taped_calls'])} unattributed call positions, {elapsed:.0f}s",
      flush=True)
print(f"wrote {a.out}", flush=True)
sys.exit(rc)
