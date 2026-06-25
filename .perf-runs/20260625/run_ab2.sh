#!/bin/bash
# Usage: run_ab2.sh <size> <mode> <traceEnvString> <card> <tag>
set -u
SIZE=$1; MODE=$2; TENV=$3; CARD=$4; TAG=$5
REPO=~/tt-bio-dev; OUT=/tmp/perf_${TAG}
rm -rf "$OUT"; mkdir -p "$OUT"
FASTFLAG=""; [ "$MODE" = "fast" ] && FASTFLAG="--fast"
cd /tmp
env TT_STAGE_PROFILE=1 $TENV PYTHONPATH=~/.tt-bio-perf/stagehook:$REPO \
    $REPO/env/bin/tt-bio predict ~/.tt-bio-perf/in${SIZE} $FASTFLAG \
    --debug --device_ids $CARD --out_dir "$OUT" --seed 0 --log > "$OUT/log.txt" 2>&1
echo "===== $TAG (size=$SIZE mode=$MODE env='$TENV' card=$CARD) ====="
grep "\[STAGEHOOK\] forward_total" "$OUT/log.txt" | tail -1 | sed 's/^/WARM: /'
