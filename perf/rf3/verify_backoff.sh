#!/usr/bin/env bash
set -u
WT=/home/ttuser/.coworker/wt/rf3-1024aa-exponent-gate
OUT=/home/ttuser/rf3_exp_work
cd "$WT" || exit 1
P(){ PYTHONPATH="$WT:/home/ttuser/rf3_perf_deps" TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 \
     TT_BIO_LEASE_HOLDER=worker:rf3-1024aa-exponent-gate \
     /home/ttuser/tt-bio-dev/env/bin/python3 "$@"; }
echo "=== fix1024 $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
P perf/rf3/trunk_decompose.py --aa 1024 --n_recycles 2 --out "$OUT/dec_fix1024.json" 2>&1 \
  | grep -E "s/recycle|tri_att|fp32_softmax \{|Error|Traceback|FATAL"
echo "=== fix768 $(date -Is) load=$(cut -d' ' -f1-3 /proc/loadavg)"
P perf/rf3/trunk_decompose.py --aa 768 --n_recycles 2 --out "$OUT/dec_fix768.json" 2>&1 \
  | grep -E "s/recycle|tri_att|fp32_softmax \{|Error|Traceback|FATAL"
echo "=== bits1024 $(date -Is)"
P perf/rf3/fp32_l1_backoff_bits.py --aa 1024 --out "$OUT/bits_1024.json" 2>&1 \
  | grep -E "^\[|_vs_|RESULT|Error|Traceback|FATAL" | head -20
echo ALLDONE_VERIFY
