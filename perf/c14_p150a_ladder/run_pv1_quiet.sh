#!/bin/bash
# protenix-v1 alone, re-recorded on a QUIET qb1. The 2026-09-18 16:41Z campaign folded it first,
# in the window where load1 ran to 31.8, and its seconds came out non-monotone (256 aa 33.0 s
# against 640 aa 26.5 s) with the exponent block skipped on a measured sigma of 47.1 %. The other
# five models folded later, on a box that had emptied, and reproduce their old qb1 rows inside 6 %.
# So this re-folds the one contended model rather than the whole ladder.
#
# benchlock.sh is the guard: it waits for the lock AND for the box to drop below loadavg 2.0 AND
# for foreign folds to stop making CPU progress. The release gate does not take the lock itself.
set -u
WT=/home/ttuser/.coworker/wt/c14-p150a-ladder-record
OUT="$WT/perf/c14_p150a_ladder/runs/quiet_pv1"
cd "$WT" || exit 1
mkdir -p "$OUT"
touch "$OUT/RUNNING"
setsid nohup python3 perf/c14_p150a_ladder/clock_watch.py \
    "$OUT/clock.jsonl" "$OUT/RUNNING" 15 >/dev/null 2>&1 < /dev/null &
date -u +"start %Y-%m-%dT%H:%M:%SZ" >> "$OUT/record.log"
BENCHLOCK_WAIT_S=2400 BENCHLOCK_LOAD_WAIT_S=1800 \
/home/ttuser/.coworker/scripts/benchlock.sh c14-p150a-ladder-record -- \
    env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
        TT_BIO_LEASE_HOLDER=worker:c14-p150a-ladder-record \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py \
    --model size-ladder --size-ladder-record --size-ladder-fragment \
    --size-ladder-models protenix-v1 \
    >> "$OUT/record.log" 2>&1
echo "EXIT=$?" >> "$OUT/record.log"
date -u +"end %Y-%m-%dT%H:%M:%SZ" >> "$OUT/record.log"
rm -f "$OUT/RUNNING"
