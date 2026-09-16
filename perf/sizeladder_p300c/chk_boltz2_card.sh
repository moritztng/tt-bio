#!/bin/bash
# boltz2 p300c size-ladder CHECK on an explicitly chosen card ($1). The baseline is keyed by card
# TYPE (p300c) and grid, not by card index, so any p300c presenting 11x10 on this host scores the
# same entry. Kept as a separate script because card 1 wedged two boltz2 folds at "trunk 0/4"
# while every esmfold2 run on that same card completed, and a sibling card is the control for
# that claim.
cd /home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
export TT_VISIBLE_DEVICES="$1" TT_BIO_LEASE_CARDS="1,$1" TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models boltz2 --keep
