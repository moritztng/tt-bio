#!/bin/bash
# The interleaved benchlocked fold A/B for TT_BIO_PAIR_Z_B8 at 512 aa, clock forced to 1350 MHz
# in-process and sampled DURING every fold. One process per arm, arms interleaved block by block.
set -u
cd /home/ttuser/.coworker/wt/bfp8-z-accumulator || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export TT_BIO_AICLK=1350
exec /home/ttuser/.coworker/scripts/benchlock.sh bfp8-z-accumulator -- \
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=0,3 TT_BIO_LEASE_HOLDER=worker:bfp8-z-accumulator \
      TT_BIO_AICLK=1350 \
  $PY perf/bfp8_z/fold_ab.py --sizes 512 --blocks 4 --folds 3 --card 3 \
      --flag TT_BIO_PAIR_Z_B8 --out perf/bfp8_z/z_ab.json --cifdir perf/bfp8_z/ab_cifs
