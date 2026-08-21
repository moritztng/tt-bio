#!/usr/bin/env bash
# At 1024 aa the fp32-softmax tail asks for a 3-row L1 shard (786432 B/core), the sharded softmax
# refuses its circular buffers around it, and the shipped fallback retires L1 for the whole shape
# class. These arms ask whether a SMALLER block would have fitted: 2 rows (524288 B/core) and
# 1 row (262144 B/core).
set -u
WT=/home/ttuser/.coworker/wt/rf3-1024aa-exponent-gate
OUT=/home/ttuser/rf3_exp_work
cd "$WT" || exit 1
run(){
  aa=$1; tag=$2; shift 2
  echo "=== $tag aa=$aa start $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg) extra=$*"
  PYTHONPATH="$WT:/home/ttuser/rf3_perf_deps" \
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-1024aa-exponent-gate \
    /home/ttuser/tt-bio-dev/env/bin/python3 perf/rf3/trunk_decompose.py \
      --aa "$aa" --n_recycles 2 --out "$OUT/dec_${tag}.json" "$@" 2>&1 \
      | grep -E "s/recycle|tri_att|fp32_softmax \{|Error|Traceback|FATAL"
  echo "=== $tag done $(date -Is)"
}
run 1024 l1_2row --fp32_l1_bytes_per_core 524288
run 1024 l1_1row --fp32_l1_bytes_per_core 262144
echo ALLDONE_AB
