#!/usr/bin/env bash
# One timed fold A/B for E6, taken only once qb2 is genuinely quiet.
set -u
WT=/home/ttuser/.coworker/wt/trimul-fused-kernel-final
cd "$WT" || exit 70
COTENANT=${COTENANT:-0}

echo "start $(date -Is) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
if [ "$COTENANT" != 0 ]; then
  i=0
  while kill -0 "$COTENANT" 2>/dev/null && [ $i -lt 80 ]; do sleep 15; i=$((i+1)); done
  if kill -0 "$COTENANT" 2>/dev/null; then
    echo "ABORT: co-tenant pid $COTENANT still folding after $((i*15))s; not measuring"
    exit 75
  fi
  echo "co-tenant pid $COTENANT gone at $(date -Is) after $((i*15))s"
fi
echo "ps check:"; ps -eo pid,etime,cmd | grep -E "fold_ab512|tt_concurrency" | grep -v grep

BENCHLOCK_MAXLOAD=1.0 BENCHLOCK_LOAD_WAIT_S=600 \
/home/ttuser/.coworker/scripts/benchlock.sh trimul-fused-kernel-final -- \
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:trimul-fused-kernel-final \
      PYTHONPATH="$WT" \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/size512/fold_ab512.py \
    --sizes 512 --arms on,e6,on,e6 \
    --out perf/trimul_f2/fold_e6_512_qb2c0.json
rc=$?
echo "done $(date -Is) rc=$rc loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
exit $rc
