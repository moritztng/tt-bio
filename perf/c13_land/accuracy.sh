#!/usr/bin/env bash
# Accuracy re-score for the shipped stack on the merged tree: base vs both at two seeds, with the
# same-seed A/A control and the seed-scatter floor measured in the same session on the same fixture.
# 298 aa is the cell the 0.35/0.60 A bar is written against; 512 aa is reported with plDDT deciding,
# because its unconstrained hinge saturates whole-molecule RMSD for any non-bit-exact change.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c13_land
cd "$WT" || exit 1
for sz in 298 512; do
  echo "=== ACC $sz $(date -Is) ==="
  /home/ttuser/.coworker/scripts/benchlock.sh c13-land-first -- env \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
    "$PY" perf/c12_compose/acc_stack.py \
      --out "$OUT/acc${sz}.json" --cifs "$OUT/acc${sz}_cifs" \
      --size "$sz" --seeds 0,1 --arms base,both --mhz 1350
  echo "rc=$? size=$sz"
done
echo "=== ACC DONE $(date -Is) ==="
