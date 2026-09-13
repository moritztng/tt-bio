#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/util-op-deletes
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:util-op-deletes
for F in 0 1; do
  echo "=== $(date -u +%H:%M:%S) flag=$F"
  ( cd "$WT" && PYTHONPATH="$WT" TT_BIO_TRIMUL_MM_TRANSPOSE=$F timeout 600 \
    /home/ttuser/tt-bio-dev/env/bin/python3 perf/util_op_deletes/flag_fired.py \
    --phases control --reps 1 --size 512 --out /home/ttuser/scratch/uod/fired_$F.json \
    --cifdir /home/ttuser/scratch/uod/firedcif_$F ) 2>&1 | grep -E "FIRED|control|Error|Traceback" | tail -4
done
echo "=== $(date -u +%H:%M:%S) FIRED DONE"
