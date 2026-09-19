#!/usr/bin/env bash
# The post-merge fold A/B for the shipped stack: silu + cond-hoist, interleaved rep by rep,
# benchlocked, clock forced and sampled DURING each fold, with the session's own A/A floor from
# base at both ends of every rep.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c13_land
cd "$WT" || exit 1
exec /home/ttuser/.coworker/scripts/benchlock.sh c13-land-first -- env \
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
  "$PY" perf/c12_compose/fold_compose.py \
    --out "$OUT/remeasure512.json" --cifs "$OUT/remeasure512_cifs" \
    --arms base,silu,hoist,both,base --reps 5 --size 512 --mhz 1350 --palindrome
