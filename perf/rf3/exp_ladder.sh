#!/usr/bin/env bash
# Per-op trunk decomposition across the 512/768/1024 rungs on the shipped RF3 default (no levers),
# so each op class gets its own local log-log exponent across the 768 -> 1024 interval.
# Arm order 768, 1024, 512, 768-again: the repeat bounds A/A drift on the interval that matters.
set -u
WT=/home/ttuser/.coworker/wt/rf3-1024aa-exponent-gate
OUT=/home/ttuser/rf3_exp_work
mkdir -p "$OUT"
cd "$WT" || exit 1
run(){
  aa=$1; tag=$2
  echo "=== $tag aa=$aa start $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
  PYTHONPATH="$WT:/home/ttuser/rf3_perf_deps" \
  TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:rf3-1024aa-exponent-gate \
    /home/ttuser/tt-bio-dev/env/bin/python3 perf/rf3/trunk_decompose.py \
      --aa "$aa" --n_recycles 2 --out "$OUT/dec_${tag}.json" 2>&1 | tail -90
  echo "=== $tag done $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
}
run 768 a768
run 1024 a1024
run 512 a512
run 768 b768
echo ALLDONE
