#!/bin/bash
# The 44-leg implementation-parity gate for wk/b2z2-zinit-ship.
# Resumable per leg: re-running with the same --workdir picks up where it stopped.
set -u
W=/home/ttuser/.coworker/wt/b2z2-zinit-ship
cd $W
export PYTHONPATH=$W
export OPENDDE_DOCKQ_PYTHON=/home/ttuser/dockqenv/bin/python3
export TT_BIO_LEASE_HOLDER=worker:b2z2-zinit-ship
export TT_BIO_LEASE_CARDS=1
/home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/full_parity_gate.py \
  --workers localhost:1 \
  --workdir $W/perf/b2z2_gate/zinitship_rebased_work \
  --out $W/perf/b2z2_gate/gate_zinitship_rebased.json
echo GATEDONE
