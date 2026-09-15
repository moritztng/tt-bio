#!/usr/bin/env bash
# N repeats of one arm, one fold per process, NO trace env, so the shipped path is what is measured.
#
# The wedge is intermittent, so a single completion is not evidence that anything fixed it and a
# single hang is not evidence that an arm is broken. What separates the two is a RATE, measured on
# both arms under the same box conditions -- which is why the caller runs one arm per card at the
# same time rather than one arm after the other.
#
# A wedged fold is detected by wall clock alone: 1024 aa folds in ~55 s on this box and the wedge
# holds the host at 100 % CPU indefinitely, so anything past the cap is the wedge. No trace hook is
# involved, so the instrument cannot perturb the race it counts.
#
# The fold runs under `setsid`, so killing a wedged one kills the whole process group. tt_bio forks
# a worker child that holds the device; killing only the outer pid leaves that child on the card and
# every later repeat fails its device open and writes no record at all, which is how the first run
# of this harness produced 2 of 6 repeats and looked like the script had crashed.
set -u
WT=/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix
OUT=$WT/perf/roof_transition_l1_1024/out
SIZE=${SIZE:-1024}
ARM=${ARM:-on}
REPS=${REPS:-3}
CAP=${CAP:-150}
CARD=${CARD:-3}
LEASE=${LEASE:-$CARD}
STEPS=${STEPS:-2}
SUF=${SUF:-}
mkdir -p "$OUT"
cd "$WT" || exit 2
for i in $(seq 1 "$REPS"); do
  TAG=rep${SUF}_${SIZE}_${ARM}_c${CARD}_$i
  LOG=$OUT/$TAG.log
  : > "$LOG"
  rm -f "$OUT/$TAG.json"
  T0=$(date +%s)
  setsid env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$LEASE \
    TT_BIO_LEASE_HOLDER=worker:roof-transition-l1-1024-overflow-fix \
    /home/ttuser/tt-bio-dev/env/bin/python3 perf/roof_transition_l1_1024/hfold.py \
    --size "$SIZE" --arm "$ARM" --steps "$STEPS" --out "$OUT/$TAG.json" >>"$LOG" 2>&1 &
  PID=$!
  WEDGE=0
  while kill -0 "$PID" 2>/dev/null; do
    sleep 5
    if [ $(( $(date +%s) - T0 )) -ge "$CAP" ]; then
      WEDGE=1
      kill -9 -"$PID" 2>/dev/null || kill -9 "$PID" 2>/dev/null
      sleep 5
      break
    fi
  done
  wait "$PID" 2>/dev/null
  if [ "$WEDGE" = 1 ]; then
    echo "REP $SIZE $ARM card$CARD #$i WEDGED >${CAP}s throws=$(grep -c TT_THROW "$LOG")"
  else
    echo "REP $SIZE $ARM card$CARD #$i $(grep -h '^HFOLD' "$LOG" | tail -1) throws=$(grep -c TT_THROW "$LOG")"
  fi
done
echo "REPS-END $SIZE $ARM card$CARD"
