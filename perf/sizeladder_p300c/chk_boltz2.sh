#!/bin/bash
# boltz2 p300c size-ladder CHECK, full ladder in one process. boltz2's baseline was recorded as a
# single six-rung pass, so its exponents are internally consistent and its check must be too: an
# exponent is a ratio between two rungs and mixing passes gives a ratio neither pass measured.
# The whole boltz2 ladder is ~7 min of folds, which fits a qb2 window; esmfold2's does not, which
# is why that model is checked in slices with the gated rungs kept together.
cd /home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-models boltz2 --keep
