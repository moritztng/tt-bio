#!/bin/bash
# The fold A/B for TT_BIO_DEVICE_CONDITIONING, under benchlock so it cannot be co-tenanted.
#
# `b2z2-union-land` could not time this lever at all: a sibling three-card gate had qb2 at loadavg
# 29-42 and 512 aa folds scattered 20.987-28.252 s. Same situation here, so this waits rather than
# publishing noise. benchlock takes the lock, then waits for no foreign fold and for the box to be
# genuinely quiet, and only then starts the clock.
set -u
cd /home/ttuser/.coworker/wt/b2z2-cond-ship || exit 1

export BENCHLOCK_WAIT_S=3300
export BENCHLOCK_LOAD_WAIT_S=3300

exec /home/ttuser/.coworker/scripts/benchlock.sh b2z2-cond-ship -- \
  env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:b2z2-cond-ship \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cond/fold_cond.py \
    --out perf/b2z2_cond/out/timing_qb2c1.json \
    --cifdir perf/b2z2_cond/cif_timing \
    --sizes 512 \
    --timing-reps 5 \
    --timing-arms base,cond,base \
    --timing-seed 0
