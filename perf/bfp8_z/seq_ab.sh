#!/bin/bash
# The tighter A/B: arms alternated FOLD BY FOLD in one process, so model load, program cache and
# host drift are common-mode and every base fold has an on fold beside it. `_PAIR_Z_B8` is read at
# call time inside Pairformer.__call__, so it can be flipped in process; the two warmup folds at
# the head mean no timed fold carries a program-cache miss.
set -u
cd /home/ttuser/.coworker/wt/bfp8-z-accumulator || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
export BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-150}
exec /home/ttuser/.coworker/scripts/benchlock.sh bfp8-z-accumulator -- \
  env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=0,3 TT_BIO_LEASE_HOLDER=worker:bfp8-z-accumulator \
  $PY perf/bfp8_z/fold_z.py --out perf/bfp8_z/z_seq_ab.json --cifdir perf/bfp8_z/seq_cifs \
      --sizes 512 --aiclk 1350 --seed 0 \
      --seq base,Z,base,Z,base,Z,base,Z,base,Z,base,Z
