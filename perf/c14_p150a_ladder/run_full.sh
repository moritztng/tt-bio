#!/bin/bash
# Full p150a size-ladder record, six models, detached. Fragments land per model, so a pass
# that dies mid-campaign keeps every model it already finished.
set -u
WT=/home/ttuser/.coworker/wt/c14-p150a-ladder-record
OUT="$WT/perf/c14_p150a_ladder/runs/full"
cd "$WT" || exit 1
mkdir -p "$OUT"
touch "$OUT/RUNNING"
setsid nohup python3 perf/c14_p150a_ladder/clock_watch.py \
    "$OUT/clock.jsonl" "$OUT/RUNNING" 20 >/dev/null 2>&1 < /dev/null &
date -u +"start %Y-%m-%dT%H:%M:%SZ" >> "$OUT/record.log"
TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
TT_BIO_LEASE_HOLDER=worker:c14-p150a-ladder-record \
    /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py \
    --model size-ladder --size-ladder-record --size-ladder-fragment \
    --size-ladder-models protenix-v1,boltz2,nesso1,openbind,opendde,openfold3 \
    >> "$OUT/record.log" 2>&1
echo "EXIT=$?" >> "$OUT/record.log"
date -u +"end %Y-%m-%dT%H:%M:%SZ" >> "$OUT/record.log"
rm -f "$OUT/RUNNING"
