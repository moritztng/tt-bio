#!/bin/bash
# The Blackhole axis sweep, 288 to 864 in 32-token steps, at the real model (hHSA-class
# targets, binder chosen so the complex lands mid-bucket; the seam axis is read back from
# rung.json). Two rounds a rung. usage: sweep.sh <card> <tree> "<target:binder:axis> ..."
set -u
card=$1 tree=$2; shift 2
wt=/home/ttuser/.coworker/wt/bcw-bmm
out=$wt/perf/bcw_bmm/out/sweep; mkdir -p $out
log=$out/card$card.log
cd $tree
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2 JAX_COMPILATION_CACHE_DIR=$out/xlacache_$card
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=0,$card TT_BIO_LEASE_HOLDER=worker:bcw-bmm
for r in $1; do IFS=: read t b ax <<< "$r"
  tag=a${ax}_${t}_${b}$([ "$tree" = "$wt" ] || echo _main)
  echo "=== $(date -u +%FT%TZ) $tag tree=$tree commit=$(git -C $wt rev-parse --short HEAD)" >> $log
  timeout 2400 /home/ttuser/bcx_e2e_venv/bin/python3 -u $wt/perf/bcw_bmm/rung_narrowed.py \
    --params /home/ttuser/bcx_e2e/af2_params --out $out/$tag --target $t --binder $b \
    --rounds 2 --trajectories 1 > $out/$tag.log 2>&1
  echo "=== rc=$? $(date -u +%FT%TZ) $tag" >> $log
done
echo "=== SWEEP DONE $(date -u +%FT%TZ)" >> $log
