#!/usr/bin/env bash
# Lever B end-to-end A/B on the pinned rfd3_R4 fixture: RFD3_TUNE_MATMUL=1 against shipped
# defaults. Arms interleaved A B A B under ONE benchlock hold, so the two A runs are also the A/A
# floor. 40 timesteps at the fixture's production batch of 2; the metric is the median warm step,
# which excludes the 9.8 s first-step compile tail by construction.
set -u
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/p42/ab_tune
mkdir -p "$OUT"
for rep in 1 2; do
  for arm in A B; do
    if [ "$arm" = B ]; then EXTRA="RFD3_TUNE_MATMUL=1"; else EXTRA="RFD3_TUNE_MATMUL=0"; fi
    echo "=== arm=$arm rep=$rep $(date -Is) ==="
    env $EXTRA TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture \
        PYTHONPATH=$PWD "$PY" scripts/rfd3_port/p42_drain_attribution.py \
        --num_timesteps 40 --designs 2 --out "$OUT/${arm}${rep}.json" 2>&1 \
      | grep -E "^\[drain\]|^\[done\]|Error|Traceback"
  done
done
echo "=== ALL DONE $(date -Is) ==="
