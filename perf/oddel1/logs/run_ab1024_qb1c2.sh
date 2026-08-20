#!/bin/sh
# Defect A GO leg (state doc 5.3 / 16.1) on qb1 card 2: a CLEAN 13x10 part, 130 cores.
# pc card 0 is the only other 13x10 part on the fleet and section 14 disqualified it for the
# byte-identity arm (two `on` legs, two different CIF hashes). Four interleaved legs so both
# arms get their own A/A.
WT=/home/ttuser/.coworker/wt/opendde-size-generality-l1-work-split
cd $WT || exit 1
export TT_VISIBLE_DEVICES=2
export TT_BIO_LEASE_HOLDER=worker:opendde-size-generality-l1-work-split
export PYTHONPATH=$WT
# Wait for the section 6.1 run on the same card to release it. Poll existence, bounded, and
# require CPU progress so a wedged pid cannot hold this off forever (defer-loop-polls-a-wedged-pid).
PREV_PAT="fold_ab_multi.py --model opendde --sizes 640,768"
i=0
while [ $i -lt 240 ]; do
  pid=$(pgrep -f "$PREV_PAT" | head -1)
  [ -z "$pid" ] && break
  t0=$(awk "{sub(/^.*\) /,\"\"); print \$12+\$13}" /proc/$pid/stat 2>/dev/null)
  sleep 20
  t1=$(awk "{sub(/^.*\) /,\"\"); print \$12+\$13}" /proc/$pid/stat 2>/dev/null)
  if [ -n "$t0" ] && [ -n "$t1" ] && [ $((t1 - t0)) -lt 2 ]; then
    echo "chain: pid $pid burned no CPU in 20 s, treating the card as free" >&2
    break
  fi
  i=$((i+1))
done
echo "chain: starting the 1024 aa A/B at $(date -Is), loadavg $(cut -d\  -f1-3 /proc/loadavg)" >&2
BENCHLOCK_LOAD_WAIT_S=600 exec /home/ttuser/.coworker/scripts/benchlock.sh \
  opendde-size-generality-l1-work-split -- \
  /home/ttuser/tt-bio-dev/env/bin/python3 -u perf/other512/fold_ab_multi.py \
  --model opendde --sizes 1024 --arms on,qpercore,on,qpercore \
  --out perf/oddel1/fold_ab_1024_qb1c2.json
