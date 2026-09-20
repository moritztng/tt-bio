#!/bin/bash
# Phase 8 outer -- queue for the box, and keep queueing until the deadline.
#
# The defect this fixes in phase 6 is the attempt COUNT. Phase 6 ran a fixed list of six tags and
# gave up when the list ran out, which on a box carrying a foreign host job means it can burn every
# attempt inside one contaminated hour and exit at 06:00Z with a deadline of 16:00Z it never
# reached. The resource this row is short of is a quiet window, not attempts, so the loop is bounded
# by the clock alone.
#
# Where the admissibility test lives is settled and unchanged from phase 6: BLOCK on the lock,
# because the queue is what benchlock is for and a box with a legitimate holder folding on it never
# reads quiet from outside. Decide INSIDE the lock, where every other benchlock user is excluded and
# the load that remains is exactly the contamination that matters, and release in seconds if it is
# too high rather than squatting (the phase 4 failure) .
set -u
OUT=${OUT:-/home/ttuser/pvx_qb2/out3}
BL=${BL:-/home/ttuser/.coworker/scripts/benchlock.sh}
INNER=${INNER:-/home/ttuser/pvx_qb2/p8inner.sh}
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
DEADLINE=${DEADLINE_EPOCH:-$(( $(date +%s) + 36000 ))}
BACKOFF=${BACKOFF:-420}
WANT=${WANT:-2}
CELLS="ctl_open b2_old ptx_p300c b2_pairnew"
mkdir -p "$OUT"
echo "=== $(date -u +%FT%TZ) phase 8, deadline $(date -u -d @"$DEADLINE" +%FT%TZ), want $WANT accepted ==="

summarised(){ [ -s "$1" ] && grep -q "\"summary\"" "$1" 2>/dev/null; }
median(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))[\"summary\"][\"median_fold_s\"])" "$1" 2>/dev/null; }
field(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))[\"summary\"].get(sys.argv[2]))" "$1" "$2" 2>/dev/null; }
park(){ [ -e "$1" ] || return 0; local d="${1%.json}_PARKED_$(date -u +%H%M%SZ).json"; mv "$1" "$d"; echo "    parked $(basename "$d")"; }
parkall(){ for c in $CELLS; do park "$OUT/${c}_$1.json"; done; }

attempt(){
  local t=$1 left
  parkall "$t"
  left=$(( DEADLINE - $(date +%s) )); [ "$left" -gt 900 ] || return 2
  echo "=== $(date -u +%H:%M:%SZ) attempt $t, queueing for the lock (budget ${left}s) ==="
  TAG=$t BENCHLOCK_WAIT_S=$left BENCHLOCK_LOAD_WAIT_S=30 "$BL" pvx-baseline -- bash "$INNER"
  local rc=$?
  if [ "$rc" != 0 ]; then
    echo "    attempt $t: rc=$rc (75 box too loaded or lock timeout, 76 opening control, 77 closing control)"
    parkall "$t"; return 1
  fi
  # A zero rc from the inner means both controls held. Still refuse the board-pair power budget,
  # which is a physical fact about the part and not a proxy for anything.
  for c in b2_old ptx_p300c b2_pairnew; do
    summarised "$OUT/${c}_$t.json" || { echo "    $c produced no summary"; parkall "$t"; return 1; }
    if [ "$(field "$OUT/${c}_$t.json" partner_busy)" = "True" ]; then
      echo "    $c: the board partner was drawing power, shared budget -- parking"; parkall "$t"; return 1
    fi
  done
  local o n p
  o=$(median "$OUT/b2_old_$t.json"); n=$(median "$OUT/b2_pairnew_$t.json"); p=$(median "$OUT/ptx_p300c_$t.json")
  echo "=== $(date -u +%H:%M:%SZ) $t ACCEPTED: boltz2 old $o / new $n = $(awk -v a="$o" -v b="$n" "BEGIN{printf \"%.4f\", a/b}")x, protenix-v2 $p s ==="
  return 0
}

ok=0; i=0
while [ "$ok" -lt "$WANT" ]; do
  i=$((i+1))
  attempt "s$i"; rc=$?
  [ "$rc" = 2 ] && { echo "=== deadline reached with $ok accepted ==="; break; }
  [ "$rc" = 0 ] && { ok=$((ok+1)); continue; }
  sleep "$BACKOFF"
done
echo "PVXBASELINEP8DONE $(date -u +%FT%TZ) accepted=$ok"
