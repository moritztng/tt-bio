#!/bin/bash
# esmfold2 p300c size-ladder CHECK, one slice of the ladder per process. $1 = card, $2 = rungs.
# Check mode writes no baseline, so two slices can run on two cards at once; the timing-derived
# exponent intervals (256->512, 512->768) are kept inside ONE slice so a ratio is never taken
# across two differently loaded processes.
cd /home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
export TT_VISIBLE_DEVICES="$1" TT_BIO_LEASE_CARDS="1,$1" TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh
export RELEASE_GATE_SIZE_RUNGS="$2"
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models esmfold2 --keep
