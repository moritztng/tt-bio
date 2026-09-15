#!/usr/bin/env bash
# Run one foldab leg with the Transition trace on, a hard wall-clock cap, and a stall detector.
# Prints the last trace line and a WEDGED/DONE/THROW verdict, so a leg is classified without a
# human reading 20k lines of log.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix
cd "$WT" || exit 2
LEG=${1:?leg e.g. 1024:on}
TAG=${2:?tag}
CAP=${3:-420}
CARD=${4:-3}
STEPS=${5:-2}
OUT=$WT/perf/roof_transition_l1_1024/out
mkdir -p "$OUT"
LOG=$OUT/$TAG.log
: > "$LOG"
TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
TT_BIO_LEASE_HOLDER=worker:roof-transition-l1-1024-overflow-fix \
TT_BIO_TRANSITION_TRACE=1 \
  /home/ttuser/tt-bio-dev/env/bin/python3 perf/roof_transition_chunk_bh/foldab.py \
  --out "$OUT/$TAG.json" --legs "$LEG" --reps 1 --steps "$STEPS" >>"$LOG" 2>&1 &
PID=$!
echo "launched pid=$PID leg=$LEG tag=$TAG cap=${CAP}s card=$CARD steps=$STEPS"
LAST=0; STALL=0; T=0
while kill -0 "$PID" 2>/dev/null; do
  sleep 15; T=$((T+15))
  N=$(grep -c "\[transition-h\]" "$LOG" 2>/dev/null | head -1)
  N=${N:-0}
  if [ "$N" -eq "$LAST" ]; then STALL=$((STALL+15)); else STALL=0; fi
  LAST=$N
  echo "t=${T}s trace=$N stalled=${STALL}s load=$(cut -d" " -f1 /proc/loadavg)"
  if [ "$T" -ge "$CAP" ]; then
    echo "======== CAP HIT at t=${T}s, stalled=${STALL}s ========"
    kill -INT "$PID" 2>/dev/null; sleep 20; kill -9 "$PID" 2>/dev/null
    break
  fi
done
wait "$PID" 2>/dev/null; RC=$?
echo "---- rc=$RC ----"
echo "last trace: $(grep "\[transition-h\]" "$LOG" | tail -1)"
echo "throws: $(grep -c "TT_THROW" "$LOG")  absorbed: $(grep -c "device refused this L1 plan" "$LOG")"
if [ -f "$OUT/$TAG.json" ] && grep -q digest "$OUT/$TAG.json" 2>/dev/null; then echo "VERDICT=DONE"; 
elif [ "$STALL" -ge 60 ]; then echo "VERDICT=WEDGED"; else echo "VERDICT=OTHER rc=$RC"; fi
