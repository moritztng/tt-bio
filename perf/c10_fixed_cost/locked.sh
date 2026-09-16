#!/usr/bin/env bash
set -euo pipefail
OUT=${1:?output required}
shift
SIZES=("$@")
# The unchanged shared lock logs a load timeout but continues. Reject that here.
if grep -q 'benchlock: WARNING\|benchlock: TIMED OUT' "$OUT/launch.log"; then exit 75; fi
FIFO="$OUT/ambient.fifo"
mkfifo "$FIFO"
python3 perf/c10_fixed_cost/ambient.py "$OUT/ambient.jsonl" < "$FIFO" &
AMBIENT=$!
exec 9>"$FIFO"
finish() {
  echo stop >&9 2>/dev/null || true
  exec 9>&-
  wait "$AMBIENT" 2>/dev/null || true
  rm -f "$FIFO"
  [[ -f "$OUT/ambient.jsonl" ]] && gzip -n -f "$OUT/ambient.jsonl"
  return 0
}
trap finish EXIT
for SIZE in "${SIZES[@]}"; do
  TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c10-fixed-cost \
    python3 perf/c10_fixed_cost/capture.py --size "$SIZE" --out "$OUT/$SIZE"
done
