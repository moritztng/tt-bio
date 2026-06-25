#!/bin/bash
# Usage: run_ab.sh <size> <mode fast|default> <trace 0|1> <card> <tag>
# Runs a warm fold (predict on the same-size pair dir), prints the WARM (2nd) stage line.
set -u
SIZE=$1; MODE=$2; TRACE=$3; CARD=$4; TAG=$5
REPO=~/tt-bio-dev
OUT=/tmp/perf_${TAG}
rm -rf "$OUT"; mkdir -p "$OUT"
FASTFLAG=""; [ "$MODE" = "fast" ] && FASTFLAG="--fast"
TRACEENV=""; [ "$TRACE" = "1" ] && TRACEENV="TT_BIO_TRACE=1"
cd /tmp
env TT_STAGE_PROFILE=1 $TRACEENV \
    PYTHONPATH=~/.tt-bio-perf/stagehook:$REPO \
    $REPO/env/bin/tt-bio predict ~/.tt-bio-perf/in${SIZE} $FASTFLAG \
    --debug --device_ids $CARD --out_dir "$OUT" --seed 0 --log \
    > "$OUT/log.txt" 2>&1
echo "===== $TAG (size=$SIZE mode=$MODE trace=$TRACE card=$CARD) ====="
grep "\[STAGEHOOK\] forward_total" "$OUT/log.txt"
echo "--- (last STAGEHOOK above = WARM) ; exit=$? ---"
tail -3 "$OUT/log.txt" | grep -iE "error|oom|fatal|traceback" && echo "!!! ERROR DETECTED in $TAG" || true
