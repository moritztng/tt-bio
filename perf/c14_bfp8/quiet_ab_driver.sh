#!/usr/bin/env bash
# Unattended driver for the one number bfp8-region-t-land still owes: a QUIET-BOX repeat of the
# region T fold A/B. qb1 carried two live co-tenants through this row s first two passes (a
# size-ladder release gate and an abb3 training step gate), and the ledger s own caveat is that
# the original +0.6640 s was taken under co-tenancy. Measuring while loaded would reproduce the
# defect this row exists to settle, so the AB is gated behind benchlock, which waits for the box
# to actually go quiet before it starts the clock. If the box never goes quiet, benchlock exits 75
# and NOTHING is measured. That is the intended outcome, not a failure of this script.
#
# Step 1 is a correctness confirm and runs immediately: host load cannot move a digest.
set -u
cd /home/ttuser/.coworker/wt/bfp8-region-t-land || exit 1
LEASE="TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bfp8-region-t-land"

echo "=== step 1: 512 aa seed-1 digest confirm (timing-insensitive) $(date -Is) ==="
env TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
    TT_BIO_LEASE_HOLDER=worker:bfp8-region-t-land \
  python3 perf/bfp8_envelope/fold_b8.py --sizes 512 --plan "base:1;T:1" \
    --out perf/bfp8_envelope/land2_acc_s1.json \
    --cifdir perf/bfp8_envelope/land2_acc_s1_cifs 2>&1 | grep -E "^ +512 |Traceback|Error"

echo "=== step 2: quiet interleaved benchlocked fold A/B $(date -Is) ==="
BENCHLOCK_WAIT_S=10800 BENCHLOCK_LOAD_WAIT_S=10800 \
  ~/.coworker/scripts/benchlock.sh bfp8-region-t-land -- \
  env TT_BIO_AICLK=1350 TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
      TT_BIO_LEASE_HOLDER=worker:bfp8-region-t-land \
  python3 perf/c14_bfp8/fold_ab.py --sizes 512 --blocks 4 --folds 3 --card 0 \
    --flag TT_BIO_TRIATT_B8 \
    --out perf/c14_bfp8/region_t_ab_quiet.json \
    --cifdir perf/c14_bfp8/region_t_quiet_cifs
echo "=== driver exit $? at $(date -Is) ==="
