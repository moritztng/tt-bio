#!/usr/bin/env bash
# The one number the region T default flip still owes: a quiet p300c A/B on CURRENT main-derived
# code. Board pair 2/3 verified clean (pair_idle --card 3: "sibling 2: no fd"); the host carries
# of3t-trunkceilings CPU census on the OTHER pair, so fold_abs per-arm quiet wait -- not a
# launch-time check -- is what decides whether anything gets timed. It refuses rather than
# proceeds, which is the point.
set -u
cd /home/ttuser/.coworker/wt/land-standing || exit 1
BENCHLOCK_WAIT_S=300 BENCHLOCK_LOAD_WAIT_S=1 \
  ~/.coworker/scripts/benchlock.sh land-standing -- \
  env TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=1,3 \
      TT_BIO_LEASE_HOLDER=worker:land-standing \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/c14_bfp8/fold_ab.py --sizes 512 --blocks 3 --folds 3 --card 3 \
    --flag TT_BIO_TRIATT_B8 --quiet-wait 1500 \
    --out perf/c14_bfp8/regiont_ab_qb2c3_current.json \
    --cifdir perf/c14_bfp8/regiont_cifs_qb2c3
echo "=== driver exit $? at $(date -Is) ==="
