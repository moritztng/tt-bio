#!/bin/bash
# One perf_regression.py invocation per model so a wedge on model N does not lose 1..N-1.
WT=/home/ttuser/.coworker/wt/perf-excellence-p150a
S=/home/ttuser/.coworker/state/perf-excellence-p150a
cd "$WT" || exit 1
for m in "$@"; do
  { echo; echo "=== $(date -u +%FT%TZ)  model=$m  card=1 (UMD 1, /dev/tenstorrent/2)  tree=$(git rev-parse --short HEAD) ==="; } >> "$S/perf.log"
  sleep 20
  q=$("$S/quiet_check.sh"); echo "quiet-check: $q" >> "$S/perf.log"
  TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:perf-excellence-p150a \
  PYTHONPATH="$WT" /home/ttuser/tt-bio-dev/env/bin/python3 scripts/perf_regression.py --model "$m" 2>&1 | tail -25 >> "$S/perf.log"
  echo "exit=${PIPESTATUS[0]}" >> "$S/perf.log"
done
echo "SWEEP-DONE $(date -u +%FT%TZ)" >> "$S/perf.log"
