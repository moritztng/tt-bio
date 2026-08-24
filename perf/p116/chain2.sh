#!/bin/sh
# p116 chain 2: both rungs of the batch-cap ladder, on the pinned v0.7.0 wheel, one process each.
# R2 (3844 atoms) is the rung the stale table in design.py calls non-monotone; R4 (6051) is the
# page fixture, whose earlier timing is confounded by the source-build runtime it was taken on.
set -e
cd /home/ttuser/.coworker/wt/rfd3-b8-to-4x-p4
PY=/home/ttuser/.coworker/rel070/relvenv/bin/python3
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1
export TT_BIO_LEASE_HOLDER=worker:rfd3-b8-to-4x-p4
for R in R2 R4; do
  echo "=== $R start $(date -u +%H:%M:%S) ==="
  $PY -u scripts/rfd3_port/p116_cap_ladder.py \
      --spec perf/dsfix/fixtures/rfd3_$R.json \
      --out perf/p116/ladder_${R}_c2.json \
      --reps 3 --warm-steps 25 > perf/p116/${R}_c2.log 2>&1 \
    && echo "=== $R done $(date -u +%H:%M:%S) ===" \
    || echo "=== $R FAILED $(date -u +%H:%M:%S) ==="
done
echo "=== chain2 complete $(date -u +%H:%M:%S) ==="
