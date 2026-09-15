#!/bin/bash
cd /home/ttuser/.coworker/wt/tt-bio-sizeladder-p300c-refresh
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:tt-bio-sizeladder-p300c-refresh
exec /home/ttuser/tt-bio-dev/env/bin/python3 -u scripts/release_gate.py --model size-ladder \
  --size-ladder-record --size-ladder-fragment --size-ladder-models esmfold2 --keep
