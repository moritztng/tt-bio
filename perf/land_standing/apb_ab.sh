#!/bin/bash
# Interleaved APB A/B on the openfold3 gate arm, at THIS tree, in one session.
# The two baselines used so far ran at a different hour; this removes that gap.
# Counters per leg, so the OFF leg is shown to have served 0 -- a control gated like its
# subject tests nothing, so the control must be shown to have MOVED.
set -u
WT=/home/ttuser/.coworker/wt/land-standing
OUT=$WT/perf/land_standing/out/apb_ab
CARD=3
mkdir -p "$OUT" /tmp/apbhook
cp "$WT/perf/land_standing/apb_sitecustomize.py" /tmp/apbhook/sitecustomize.py
cd "$WT" || exit 1
echo "TREE $(git rev-parse --short HEAD) start=$(date -u +%H:%M:%SZ)" | tee "$OUT/AB.txt"
for round in 1 2; do
  for arm in 0 1; do
    tag="r${round}_apb${arm}"
    dump="$OUT/counters_$tag.jsonl"; : > "$dump"
    t0=$(date +%s)
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:land-standing \
        PYTHONPATH="$WT:/tmp/apbhook" TT_BIO_APB_CONCAT_HEADS=$arm TT_APB_STATS_DUMP="$dump" \
        /home/ttuser/tt-bio-dev/env/bin/python3 scripts/release_gate.py --model openfold3 \
        > "$OUT/log_$tag.txt" 2>&1
    rc=$?; t1=$(date +%s)
    nums=$(grep -aE "^openfold3 " "$OUT/log_$tag.txt" | grep -aiE "pass|fail" | tail -1 | awk '{print $2" / "$3"  "$NF}')
    served=$(python3 -c "
import json,sys
t=0
for l in open('$dump'):
    v=json.loads(l).get('served') or 0
    t+=v
print(t)" 2>/dev/null)
    echo "APB=$arm round=$round rc=$rc secs=$((t1-t0)) served=$served  ->  $nums" | tee -a "$OUT/AB.txt"
  done
done
echo "AB_END" | tee -a "$OUT/AB.txt"
