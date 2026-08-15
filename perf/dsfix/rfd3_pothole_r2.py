"""The R2 batch pothole, re-measured with matmul calibration on.

At 3844 atoms the pre-calibration sweep read b=1 59.967, b=2 64.890, b=4 59.975 s/design:
non-monotone in batch at fixed size, with b=2 8 % worse than both neighbours. That was
measured before RFD3_TUNE_MATMUL became the default above 2952 atoms, and the calibrator's
_tunable gate and its program-config search both depend on the batch dimension, so the
pothole may be an artifact of the old default rather than a defect that still ships.

This drives the same end-to-end harness with two changes, both screen-only:

  * RFD3_TUNE_MATMUL=1 in the environment, so every arm is calibrated.
  * _BATCH_SPEED_CAP_ABOVE_ATOMS raised out of the way in this process, because the shipped
    cap forces batch 1 above 2952 atoms and would otherwise collapse all three arms into
    one. Nothing here writes to a shipped path.

Rows land in perf/dsfix/results/rfd3_batch_e2e.jsonl under rung "R2+pothole", so they never
collide with the shipped-default rows for the same (rung, batch).

    ~/.coworker/scripts/benchlock.sh rfd3-page-gap-rootcause -- env TT_VISIBLE_DEVICES=0 \
      TT_BIO_LEASE_HOLDER=worker:rfd3-page-gap-rootcause PYTHONPATH=$PWD RFD3_TUNE_MATMUL=1 \
      /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/dsfix/rfd3_pothole_r2.py 2,4
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.getcwd())
from tt_bio.rfd3 import design as rfd3_design  # noqa: E402

assert os.environ.get("RFD3_TUNE_MATMUL") == "1", "run with RFD3_TUNE_MATMUL=1"
rfd3_design._BATCH_SPEED_CAP_ABOVE_ATOMS = 10 ** 9

_spec = importlib.util.spec_from_file_location("rfd3_batch_e2e", "perf/dsfix/rfd3_batch_e2e.py")
_e2e = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_e2e)
assert _e2e.rfd3_design is rfd3_design, "the harness bound a different design module"

sys.argv = ["rfd3_batch_e2e.py", "R2", sys.argv[1], "+pothole"]
_e2e.main()
