#!/bin/bash
# boltz-2 p300c size-ladder CHECK, one slice of the ladder per process. $1 = card, $2 = rungs.
# The whole ladder is ~11 min of wall and qb2 watchdog-resets inside 9, so boltz-2 is now sliced
# the same way esmfold2 is: the two gated exponent intervals (256->512, 512->768) stay inside ONE
# slice so a ratio is never taken across two differently loaded processes, and 640/896/1024 feed
# lever rows only, which are timing-insensitive.
cd /home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
export TT_VISIBLE_DEVICES="$1" TT_BIO_LEASE_CARDS="${LEASE_CARDS:-$1}" TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh
export RELEASE_GATE_SIZE_RUNGS="$2"
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models boltz2 --keep
