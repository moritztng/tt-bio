#!/bin/bash
# repro.sh <out> <size> <batch> [tree]: one arm of the fair harness through repro.py, tree on PYTHONPATH.
# Defaults are the pc fair box; on a quiet box set PY, IN (specs dir), W (weights) and CARD.
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=$1 S=$2 B=$3 TREE=${4:-$(cd "$HERE/../../.." && pwd)}
PY=${PY:-$HOME/rfd4-fair/env/bin/python} IN=${IN:-$HOME/.coworker/state/rfd4/inputs} W=${W:-$HOME/rfd4-fair/bundle}
CARD=${CARD:-0}
mkdir -p "$OUT"
PYTHONPATH="$TREE" TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:rfd4-l1decline \
  timeout 3000 "$PY" "$HERE/repro.py" --spec "$IN/pdl1-$S.yaml" --weights "$W" --out "$OUT/run" \
  --part dev --batch "$B" --batches 1 --max-attempts 1 --shared-box --quiet-timeout 0 ${EXTRA:-} > "$OUT/log" 2>&1
echo "rc=$? $(date -u +%FT%TZ)" >> "$OUT/log"
