#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/roof-concat-heads-bh
cd "$WT" || exit 1
OUT=perf/roof_concat/apb_exact_qb2_p300c.json
LOG=/tmp/exact_run.log
for i in $(seq 1 30); do
  echo "=== attempt $i $(date -Is) load=$(cut -d' ' -f1 /proc/loadavg)" >> "$LOG"
  env TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 \
      TT_BIO_LEASE_HOLDER=worker:roof-concat-heads-bh PYTHONPATH="$WT" \
    timeout 900 /home/ttuser/tt-bio-dev/env/bin/python perf/roof_concat/apb_exact.py \
      --out "$OUT" --seq 320 512 768 1024 >> "$LOG" 2>&1
  grep -q '"S": 1024' "$OUT" 2>/dev/null && { echo "=== DONE $(date -Is)" >> "$LOG"; exit 0; }
  sleep 25
done
echo "=== GAVE UP $(date -Is)" >> "$LOG"
