#!/usr/bin/env bash
set -euo pipefail
OUT=${1:?output required}
shift
SIZES=("$@")
# The unchanged shared lock logs a load timeout but continues. Reject that here.
if grep -q 'benchlock: WARNING\|benchlock: TIMED OUT' "$OUT/launch.log"; then exit 75; fi
FIFO="$OUT/ambient.fifo"
mkfifo "$FIFO"
python3 perf/c10_size_scaling/ambient.py "$OUT/ambient.jsonl" < "$FIFO" &
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
  echo "=== SIZE $SIZE  $(date -u +%FT%TZ)"
  # A size that OOMs or crashes is a recorded outcome for THAT rung, not a reason to abandon the
  # rest of the ladder. The anchor below is the only hard gate.
  if ! TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:c10-size-scaling \
       python3 perf/c10_size_scaling/capture.py --size "$SIZE" --out "$OUT/$SIZE"; then
    echo "=== SIZE $SIZE FAILED (rc=$?) -- recorded, continuing the ladder"
    echo "{\"size\": $SIZE, \"failed\": true}" > "$OUT/$SIZE.failed.json"
  fi
  if [[ "$SIZE" == 512 ]]; then
    echo "=== ANCHOR GATE $(date -u +%FT%TZ)"
    python3 perf/c10_size_scaling/anchor_gate.py "$OUT" | tee "$OUT/anchor_gate.log"
  fi
done
echo "=== LADDER COMPLETE $(date -u +%FT%TZ)"
