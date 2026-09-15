#!/bin/bash
# esmfold2 p300c record, one slice of the ladder per process: qb2's watchdog resets the box
# roughly every 12 minutes under load and the full six-rung ladder needs longer than that.
# Each slice measures its own rungs fresh; the recorder carries the rungs it did not measure
# from the previous entry ONLY when commit, host and grid match, so slice 2 and 3 carry
# slice 1's real measurements and nothing from the stale 8e3a3e9e rows.
cd /home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh
export RELEASE_GATE_SIZE_RUNGS="$1"
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-record --size-ladder-fragment --size-ladder-models esmfold2 --keep
