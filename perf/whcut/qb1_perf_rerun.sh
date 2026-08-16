#!/bin/bash
# Re-run the two perf legs that failed to MEASURE on the first pass. Neither regressed:
# both spawn a local worker pool that reaches past its assigned card, and I had three jobs
# stacked on qb1s four cards, so they died on device-open. openfold3 is not re-run -- qb1
# has no OF3_CKPT and openfold3 is not in the live catalog.
#
# Waits on CARD OCCUPANCY, not on a process pattern: the resource is the thing that has to
# be free, and a pattern match also catches the status-check shells that carry it.
set -u
TREE=/home/ttuser/.coworker/wt/japanfold-wh-cutover
PY=/home/ttuser/tt-bio/env/bin/python3
cd "$TREE" || exit 1
while :; do
  BUSY=0
  for i in 0 1 2 3; do
    n=$(sudo -n fuser /dev/tenstorrent/$i 2>/dev/null | wc -w); BUSY=$((BUSY + n))
  done
  [ "$BUSY" -eq 0 ] && break
  sleep 60
done
echo "PERF RERUN START $(date -u +%FT%TZ) all four cards idle, load $(cut -d" " -f1-3 /proc/loadavg)"
env TT_VISIBLE_DEVICES=0 TT_METAL_LOGGER_LEVEL=FATAL PYTHONPATH="$TREE" \
    TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
  "$PY" scripts/perf_regression.py --model boltz2-affinity --model rfd3
echo "PERF RERUN EXIT $? $(date -u +%FT%TZ)"
