#!/usr/bin/env bash
# One hfold.py process per arm, hard-capped, killed by explicit pid. Prints one line per arm.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix
OUT=$WT/perf/roof_transition_l1_1024/out
SIZE=${SIZE:-1024}
CAP=${CAP:-300}
CARD=${CARD:-3}
STEPS=${STEPS:-2}
mkdir -p "$OUT"
cd "$WT" || exit 2
for ARM in "$@"; do
  TAG=hs_${SIZE}_${ARM}${TAGSUF:-}
  LOG=$OUT/$TAG.log
  : > "$LOG"
  TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=${LEASE:-$CARD} \
  TT_BIO_LEASE_HOLDER=worker:roof-transition-l1-1024-overflow-fix \
  TT_BIO_TRANSITION_TRACE=${TRACE:-1} \
    /home/ttuser/tt-bio-dev/env/bin/python3 perf/roof_transition_l1_1024/hfold.py \
    --size "$SIZE" --arm "$ARM" --steps "$STEPS" --out "$OUT/$TAG.json" >>"$LOG" 2>&1 &
  PID=$!
  LAST=0; STALL=0; T=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 10; T=$((T+10))
    N=$(grep -c "transition-h" "$LOG" 2>/dev/null)
    N=$(echo "$N" | head -1); N=${N:-0}
    if [ "$N" -eq "$LAST" ]; then STALL=$((STALL+10)); else STALL=0; fi
    LAST=$N
    if [ "$STALL" -ge 90 ] || [ "$T" -ge "$CAP" ]; then
      echo "SWEEP $SIZE $ARM WEDGED t=${T}s stalled=${STALL}s trace=$N last=\"$(grep transition-h "$LOG" | tail -1 | cut -c1-140)\""
      kill -9 "$PID" 2>/dev/null
      sleep 3
      break
    fi
  done
  wait "$PID" 2>/dev/null
  grep -h "^HFOLD" "$LOG" 2>/dev/null || true
  echo "SWEEP $SIZE $ARM end t=${T}s throws=$(grep -c TT_THROW "$LOG") absorbed=$(grep -c "device refused this L1 plan" "$LOG")"
done
