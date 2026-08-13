#!/usr/bin/env bash
# Lever C isolated: does routing the Transition's silu-gated fc1 through the bit-exact calibrator
# add anything on top of lever B? Both arms run RFD3_TUNE_MATMUL=1; arm B disables the new path via
# RFD3_TUNE_ACT=0 so the only difference is the fc1 callsite. Interleaved B C B C, one hold.
set -u
cd /home/ttuser/.coworker/wt/rfd3-optimize-on-fixture || exit 1
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=perf/p42/ab_actC
mkdir -p "$OUT"
for rep in 1 2; do
  for arm in B C; do
    if [ "$arm" = C ]; then A=1; else A=0; fi
    echo "=== arm=$arm rep=$rep $(date -Is) ==="
    env RFD3_TUNE_MATMUL=1 RFD3_TUNE_ACT=$A TT_VISIBLE_DEVICES=0 \
        TT_BIO_LEASE_HOLDER=worker:rfd3-optimize-on-fixture PYTHONPATH=$PWD "$PY" \
        scripts/rfd3_port/p42_drain_attribution.py \
        --num_timesteps 30 --designs 2 --out "$OUT/${arm}${rep}.json" 2>&1 \
      | grep -E "^\[drain\]|^\[done\]|Error|Traceback"
  done
done
echo "=== ALL DONE $(date -Is) ==="
