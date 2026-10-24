#!/bin/bash
# repro.sh <out> <size> <batch> [tree]: one arm of the fair harness through repro.py, tree on PYTHONPATH.
OUT=$1 S=$2 B=$3 TREE=${4:-/home/moritz/.coworker/wt/rfd4-l1decline}
mkdir -p "$OUT"
PYTHONPATH="$TREE" TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:rfd4-l1decline \
  timeout 3000 ~/rfd4-fair/env/bin/python "$TREE/perf/rfd4/l1decline/repro.py" \
  --spec /home/moritz/.coworker/state/rfd4/inputs/pdl1-$S.yaml --weights ~/rfd4-fair/bundle --out "$OUT/run" \
  --part dev --batch "$B" --batches 1 --max-attempts 1 --shared-box > "$OUT/log" 2>&1
echo "rc=$? $(date -u +%FT%TZ)" >> "$OUT/log"
