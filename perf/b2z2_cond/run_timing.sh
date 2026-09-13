#!/bin/bash
# The fold A/B for TT_BIO_DEVICE_CONDITIONING, under benchlock so it cannot be co-tenanted.
#
# `b2z2-union-land` could not time this lever at all: a sibling three-card gate had qb2 at loadavg
# 29-42 and 512 aa folds scattered 20.987-28.252 s. Same situation here, so this waits rather than
# publishing noise. benchlock takes the lock, then waits for no foreign fold and for the box to be
# genuinely quiet, and only then starts the clock.
#
# Card 0, not an arbitrary free card. The cell this run re-cells (site/data/perf-512aa.json,
# Boltz-2 / p150a, 19.324 s) was measured on qb2, one Blackhole processor of a p300c board,
# physical card 0. Cross-session spread on that cell is 4.60 %, so the new time is only comparable
# to the old one if the card is the same. The base arm runs in this same session anyway, which is
# what the ratio is read from.
#
# 8 warm folds per arm, matching that cell's n=8. base runs at both ends of every rep, so the A/A
# floor comes out of this session rather than being quoted from another.
set -u
cd /home/ttuser/.coworker/wt/b2z2-cond-ship || exit 1

export BENCHLOCK_WAIT_S=3300
export BENCHLOCK_LOAD_WAIT_S=3300

exec /home/ttuser/.coworker/scripts/benchlock.sh b2z2-cond-ship -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=2,0 TT_BIO_LEASE_HOLDER=worker:b2z2-cond-ship \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/b2z2_cond/fold_cond.py \
    --out perf/b2z2_cond/out/timing_qb2c0.json \
    --cifdir perf/b2z2_cond/cif_timing \
    --sizes 512 \
    --timing-reps 8 \
    --timing-arms base,cond,base \
    --timing-seed 0
