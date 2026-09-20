#!/bin/bash
# Phase 10 outer -- queue for the box and keep queueing until this launch's own deadline.
#
# Same clock-bounded shape as phase 8, with a shorter backoff. Phase 8 used 420 s because its
# critical window was ~10 minutes and there was no point re-checking faster than the thing it
# needed. Phase 10's window is ~4 minutes, so a 120 s backoff costs little and catches a gap that
# phase 8 would have slept through.
set -u
OUT=${OUT:-/home/ttuser/pvx_qb2/out10}
BL=${BL:-/home/ttuser/.coworker/scripts/benchlock.sh}
INNER=${INNER:-/home/ttuser/pvx_qb2/p10inner.sh}
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
DEADLINE=${DEADLINE_EPOCH:?}
BACKOFF=${BACKOFF:-120}
CELLS="ctl_open b2_old ctl_close"
mkdir -p "$OUT"
echo "=== $(date -u +%FT%TZ) phase 10, deadline $(date -u -d @"$DEADLINE" +%FT%TZ) ==="

park(){ [ -e "$1" ] || return 0; local d="${1%.json}_PARKED_$(date -u +%H%M%SZ).json"; mv "$1" "$d"; echo "    parked $(basename "$d")"; }
parkall(){ for c in $CELLS; do park "$OUT/${c}_$1.json"; done; }

i=0
while :; do
  i=$((i+1)); t="r$i"
  parkall "$t"
  left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -lt 420 ]; then echo "=== $(date -u +%H:%M:%SZ) deadline: ${left}s left, under the 420s a pair needs ==="; break; fi
  echo "=== $(date -u +%H:%M:%SZ) attempt $t, queueing for the lock (budget ${left}s) ==="
  TAG=$t BENCHLOCK_WAIT_S=$left BENCHLOCK_LOAD_WAIT_S=30 "$BL" pvx-baseline -- bash "$INNER"
  rc=$?
  if [ "$rc" = 0 ]; then echo "PVXBASELINEP10OK $(date -u +%FT%TZ) tag=$t"; break; fi
  echo "    attempt $t: rc=$rc (75 too loaded / lock timeout, 76 opening control, 77 closing, 78 bracket drift, 79 load mismatch)"
  parkall "$t"
  sleep "$BACKOFF"
done
echo "PVXBASELINEP10END $(date -u +%FT%TZ)"
