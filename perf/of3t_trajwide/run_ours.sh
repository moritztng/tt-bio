#!/bin/bash
# The device side of the wider rung, detached. One arm at a time on card 1, ~60 min each:
#   shipped      the trajectory, on the shipped default -- no repin flag, because
#                of3t-rebind moved the re-keying into autograd.Tensor.value and the
#                default and the repaired program are now the same program
#   shipped_aa2  the same run in a SECOND process, compared bit-exactly to shipped
#   permute      break control: parameter ordering, rotated within each shape class
#   stale        break control: step k+1 reads step k-1 weights
#   norebind     break control: skip params.rebind()
#   zero         the A16 baseline, which reads exactly 1.0 by construction
set -u
cd "$(dirname "$0")/../.."
PY=/home/ttuser/tt-bio-dev/env/bin/python
L=/tmp/of3t/trajwide
mkdir -p "$L"
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:of3t-trajwide
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3
# The AICLK is sampled DURING each arm, every 60 s, not once before it. A clock read on an
# idle card is not the clock the work ran at, and a device time without one is not a
# measurement.
clock_watch() {
  while :; do
    printf "%s " "$(date -u +%FT%TZ)"
    ~/.local/bin/tt-smi -s 2>/dev/null \
      | tr -d " " | grep -i "aiclk" | head -4 | tr "\n" " "
    printf "\n"
    sleep 60
  done
}

for ARM in shipped shipped_aa2 permute stale norebind zero; do
  echo "=== $ARM start $(date -u +%FT%TZ)" >> "$L/ours_$ARM.log"
  clock_watch >> "$L/aiclk_$ARM.log" 2>&1 &
  CW=$!
  $PY perf/of3t_trajwide/trajwide.py --side ours --arm "$ARM" --threads 3 \
      >> "$L/ours_$ARM.log" 2>&1
  RC=$?
  kill "$CW" 2>/dev/null
  echo "=== $ARM done rc=$RC $(date -u +%FT%TZ)" >> "$L/ours_$ARM.log"
done
