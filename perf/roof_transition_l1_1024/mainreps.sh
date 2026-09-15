#!/usr/bin/env bash
# The same 1024 aa fold against a tree that has NO transition row-height lever in it at all.
#
# origin/main is d35f9db1f, which is this branch minus the lever merge, so if the wedge reproduces
# here it cannot be the lever and cannot be anything on wk/roof-transition-chunk-remerge-verify
# either. Extracted with git archive into /tmp so no shared checkout and no git metadata is touched.
set -u
PROBE=${PROBE:-/home/ttuser/mainprobe-roof-l1-1024}
OUT=${OUT:-/home/ttuser/.coworker/wt/roof-transition-l1-1024-overflow-fix/perf/roof_transition_l1_1024/out}
SIZE=${SIZE:-1024}
ARM=${ARM:-ship}
REPS=${REPS:-3}
CAP=${CAP:-150}
CARD=${CARD:-3}
STEPS=${STEPS:-2}
SUF=${SUF:-main}
mkdir -p "$OUT"
cd "$PROBE" || exit 2
for i in $(seq 1 "$REPS"); do
  TAG=rep${SUF}_${SIZE}_${ARM}_c${CARD}_$i
  LOG=$OUT/$TAG.log
  : > "$LOG"
  rm -f "$OUT/$TAG.json"
  T0=$(date +%s)
  setsid env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
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
    echo "MAINREP $SIZE $ARM #$i WEDGED >${CAP}s throws=$(grep -c TT_THROW "$LOG")"
  else
    echo "MAINREP $SIZE $ARM #$i $(grep -h '^HFOLD' "$LOG" | tail -1) throws=$(grep -c TT_THROW "$LOG")"
  fi
done
echo "MAINREPS-END"
