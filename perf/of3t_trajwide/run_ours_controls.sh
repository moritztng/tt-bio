#!/bin/bash
# The two control arms still owed after the layer_norm_z reference fix. Our side never reads
# the reference module, so shipped/shipped_aa2/permute/stale carry over unchanged and only
# these two were never run. Card 2, one arm at a time, ~42 min each at 1350 MHz.
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/trajwide
mkdir -p "$L"
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:of3t-trajwide
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
clock_watch() {
  while :; do
    printf "%s " "$(date -u +%FT%TZ)"
    ~/.local/bin/tt-smi -s 2>/dev/null | tr -d " " | grep -i "aiclk" | head -4 | tr "\n" " "
    printf "\n"; sleep 60
  done
}
for ARM in norebind zero; do
  echo "=== $ARM start $(date -u +%FT%TZ)" >> "$L/ours_$ARM.log"
  clock_watch >> "$L/aiclk_$ARM.log" 2>&1 &
  CW=$!
  $PY perf/of3t_trajwide/trajwide.py --side ours --arm "$ARM" --threads 3 >> "$L/ours_$ARM.log" 2>&1
  RC=$?
  kill "$CW" 2>/dev/null
  echo "=== $ARM done rc=$RC $(date -u +%FT%TZ)" >> "$L/ours_$ARM.log"
done
