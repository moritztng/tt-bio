#!/usr/bin/env bash
# Accuracy re-score on the SHIPPED configuration: cond-hoist ALONE, not the silu+hoist stack.
# The stack reading (base vs both) no longer describes what ships -- _UNFUSED_SILU was withdrawn
# on Moritz's standing Protenix-v2 CA-lDDT refusal, so hoist-only is the shipped arm.
set -u
WT=/home/ttuser/.coworker/wt/c13-land-first
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=$WT/perf/c13_land
cd "$WT" || exit 1
echo "HEAD $(git rev-parse --short HEAD)"
grep -n "^_UNFUSED_SILU\|^_B2_DIT_COND_HOIST" tt_bio/tenstorrent.py
for sz in 298 512; do
  echo "=== ACC-HOIST $sz $(date -Is) ==="
  /home/ttuser/.coworker/scripts/benchlock.sh c13-land-first -- env \
    TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c13-land-first \
    "$PY" perf/c12_compose/acc_stack.py \
      --out "$OUT/acc${sz}_hoist.json" --cifs "$OUT/acc${sz}_hoist_cifs" \
      --size "$sz" --seeds 0,1 --arms base,hoist --mhz 1350
  echo "rc=$? size=$sz"
done
echo "=== ACC-HOIST DONE $(date -Is) ==="
