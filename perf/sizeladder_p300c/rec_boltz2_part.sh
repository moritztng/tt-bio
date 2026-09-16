#!/bin/bash
# boltz-2 p300c size-ladder RECORD, one slice of the ladder per process. $1 = card, $2 = rungs.
# The two gated exponent intervals are 256->512 and 512->768, and an exponent is a RATIO between
# two rungs, so those three rungs must be measured in ONE process: _size_ladder_carry_rungs gates
# the carry on commit, host and grid and NOT on load, so a carried endpoint contributes a number
# measured under different load and the ratio belongs to no pass. 640/896/1024 feed lever rows
# only, which are code-path facts and carry fine, so they may be sliced one rung at a time --
# which is what makes this recordable on a box that watchdog-resets every 5-15 minutes.
# HEAD must not move between slices or the carry refuses and the entry keeps only its own rungs.
cd /home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
export TT_VISIBLE_DEVICES="$1" TT_BIO_LEASE_CARDS="$1" TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh
export RELEASE_GATE_SIZE_RUNGS="$2"
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-record --size-ladder-fragment --size-ladder-models boltz2 --keep
