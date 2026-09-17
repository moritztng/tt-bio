#!/usr/bin/env bash
# Detached launcher for the accuracy run: ssh must not hold the job open.
set -u
cd /home/ttuser/.coworker/wt/c12-compose-fold || exit 1
SIZE="${1:?size}"
setsid nohup env TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
  TT_BIO_LEASE_HOLDER=worker:c12-compose-fold \
  /home/ttuser/scratch/i14venv/bin/python3 perf/c12_compose/acc_stack.py \
  --out "perf/c12_compose/out/acc${SIZE}.json" \
  --cifs "perf/c12_compose/out/acc${SIZE}_cifs" \
  --size "$SIZE" --seeds 0,1 --arms base,both \
  > "perf/c12_compose/out/acc${SIZE}.log" 2>&1 < /dev/null &
echo "launched acc${SIZE} pid=$!"
