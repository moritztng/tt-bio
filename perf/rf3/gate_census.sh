#!/usr/bin/env bash
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
      --aa "$aa" --n_recycles 2 --out "$OUT/dec_${tag}.json" "$@" 2>&1 | grep -E "^[0-9]+ aa|^  |fp32_softmax|Error|Traceback|FATAL" | head -30
  echo "=== $tag done $(date -Is)"
}
run 768 s768
run 1024 s1024
echo ALLDONE_CENSUS
