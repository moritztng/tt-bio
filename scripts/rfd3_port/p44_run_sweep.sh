#!/bin/sh
# Both arms x 2 interleaved rounds x {batch 1, batch 8} at the five doc rows, one benchlock
# hold per point so co-tenants can interleave between points rather than waiting out the lot.
WT=/home/ttuser/.coworker/wt/rfd3-host-half-defaults-on
cd "$WT" || exit 1
for p in 0 1 2 3 4; do
  BENCHLOCK_WAIT_S=7200 BENCHLOCK_LOAD_WAIT_S=600 \
  /home/ttuser/.coworker/scripts/benchlock.sh rfd3-host-half-defaults-on -- \
    env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_HOLDER=worker:rfd3-host-half-defaults-on \
      PYTHONPATH="$WT" /home/ttuser/tt-bio-dev/env/bin/python3 \
      scripts/rfd3_port/p44_throughput_table.py --out perf/p44/throughput.jsonl --points "$p"
  echo "=== point $p finished rc=$? $(date -Is) ==="
done
echo "SWEEP DONE"
