#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/rf3-1024aa-exponent-gate
OUT=/home/ttuser/rf3_exp_work
cd "$WT" || exit 1
P(){ PYTHONPATH="$WT:/home/ttuser/rf3_perf_deps" TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
     TT_BIO_LEASE_HOLDER=worker:rf3-1024aa-exponent-gate \
     /home/ttuser/tt-bio-dev/env/bin/python3 "$@"; }
for aa in 768 1024; do
  echo "=== latch$aa $(date -Is)"
  P perf/rf3/trunk_decompose.py --aa $aa --n_recycles 2 --out "$OUT/dec_latch$aa.json" 2>&1 \
    | grep -E "s/recycle|fp32_softmax \{|latched|Error|Traceback|FATAL"
done
echo ALLDONE_LATCH
