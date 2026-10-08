#!/bin/bash
# One bench.py arm on one chip with a stall watchdog, for counting device hangs.
#   perf/spd/hangwatch.sh CHIP OUT ARM INPUTS STALL_S [bench.py args...]
#   perf/spd/hangwatch.sh 14 ~/spd/spd-fasthang/h1/c730 fast:fast c730 1200 --warm 9 --share 32
# bench.py appends one record to OUT/bench.jsonl per finished fold. If that file has not grown for STALL_S
# seconds the fold is called hung: py-spy dumps every thread (OUT/hang.pyspy), then SIGINT, then SIGTERM
# 120 s later. A hung chip is left guarded: a `sleep` holds its flock and OUT/HUNG names it, so nobody opens it
# before spd-galaxy resets it. Stall time and the fold it hung in go to OUT/run.log.
set -u
CHIP=$1 OUT=$2 ARM=$3 INPUTS=$4 STALL=$5; shift 5
ROW=${TT_BIO_LEASE_HOLDER:-spd-fasthang}
PYSPY=${PYSPY:-/tmp/pyspy/bin/py-spy}
mkdir -p "$OUT" ~/spd/locks ~/spd/$ROW/leases
[ -f ~/japanfold/env.sh ] && . ~/japanfold/env.sh > /dev/null 2>&1
cd "$(dirname "$0")/../.."
export PYTHONPATH=$PWD TT_VISIBLE_DEVICES=$CHIP TT_BIO_LEASE_HOLDER=$ROW TT_BIO_LEASE_DIR=$HOME/spd/$ROW/leases
LOCK=${LOCK:-$HOME/spd/locks/chip$CHIP.lock}
say() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/run.log"; }
exec 9> "$LOCK"
say "chip $CHIP waiting for flock"
flock -w "${WAIT:-600}" 9 || { say "flock busy, stopping"; exit 3; }
say "start chip $CHIP head $(git rev-parse --short HEAD) arm $ARM inputs $INPUTS stall ${STALL}s args $*"
"${PY:-python}" perf/spd/bench.py --out "$OUT/b" --chip "$CHIP" --arm "$ARM" --inputs "$INPUTS" "$@" \
  > "$OUT/bench.log" 2>&1 &
pid=$!
J=$OUT/b/bench.jsonl
size() { stat -c %s "$J" 2>/dev/null || echo 0; }
last=$(size) t_last=$(date +%s)
while kill -0 $pid 2>/dev/null; do
  sleep 20
  s=$(size)
  if [ "$s" != "$last" ]; then last=$s t_last=$(date +%s); continue; fi
  idle=$(( $(date +%s) - t_last ))
  [ $idle -lt "$STALL" ] && continue
  folds=$(grep -c '"fold_s"' "$J" 2>/dev/null)
  say "HANG pid $pid: no fold record for ${idle}s after $folds folds; cpu $(ps -o pcpu= -p $pid)"
  timeout 120 "$PYSPY" dump --native --pid $pid > "$OUT/hang.pyspy" 2>&1
  timeout 120 "$PYSPY" dump --pid $pid > "$OUT/hang.py.pyspy" 2>&1
  kill -INT $pid; for i in $(seq 24); do sleep 5; kill -0 $pid 2>/dev/null || break; done
  kill -0 $pid 2>/dev/null && { say "SIGINT ignored, SIGTERM"; kill -TERM $pid; }
  wait $pid; rc=$?
  echo "chip $CHIP hung $(date -u +%FT%TZ) under $ROW; reset before reuse" > "$OUT/HUNG"
  say "rc=$rc chip $CHIP left guarded (flock held) for reset"
  exec sleep 43200
done
wait $pid; rc=$?
say "end rc=$rc folds $(grep -c '"fold_s"' "$J" 2>/dev/null)"
