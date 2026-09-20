#!/bin/bash
# Phase 6 -- queue for the lock, then decide inside it. Phases 4 and 5 failed in opposite
# directions and both failures are about WHERE the admissibility test lives.
#
#   Phase 4 waited for quiet INSIDE the lock, so it starved the queue (two rows, 23 min, this
#           morning) and then banked a suspect session anyway when its quiet-wait expired.
#   Phase 5 waited for quiet OUTSIDE the lock, at loadavg <= 2.0. But a box running a legitimate
#           benchlock holder never reads 2.0, so phase 5 could not tell "another row is folding,
#           which my turn in the queue resolves" from "a job that ignores the lock is running,
#           which it does not". It waited 7 minutes for a condition that cannot occur and would
#           have waited until 16:00Z.
#
# The queue is what benchlock is FOR, so block on it. Once held, every other benchlock user is
# excluded and the remaining load is exactly the contamination that matters. Probe it there and
# release within seconds if it is too high.
#
# ACCEPTANCE IS THE CONTROL, NOT clean_session. cell.py's clean_session is a conjunction that
# includes host_quiet (load_max_during <= 3.0), which is a PROXY for "the fold was not slowed".
# The pair carries the direct measurement of the same thing: the main arm reads 14.360 s on this
# card (c14-land-tail's independent 36-fold median: 14.308 s), so if it comes back inside 6 % the
# window was good whatever loadavg said, and if it does not the pair is parked whatever
# loadavg said. This pass measured that arm at 24.023 s under load, so the control has real teeth.
# partner_busy is still required false -- that one is the shared power budget, not a proxy.
set -u
OUT=/home/ttuser/pvx_qb2/out3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
INNER=/home/ttuser/pvx_qb2/p6inner.sh
NEW_REF=14.360
NEW_TOL=0.06
ENTER_MAX=${ENTER_MAX:-10.0}
DEADLINE=${DEADLINE_EPOCH:-$(( $(date +%s) + 36000 ))}
BACKOFF=${BACKOFF:-300}
mkdir -p "$OUT"
echo "=== $(date -u +%FT%TZ) phase 6, deadline $(date -u -d @$DEADLINE +%FT%TZ), enter<=$ENTER_MAX, control $NEW_REF s +/-$(awk -v t=$NEW_TOL 'BEGIN{print t*100}') % ==="

summarised(){ [ -s "$1" ] && grep -q '"summary"' "$1" 2>/dev/null; }
median(){ python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["summary"]["median_fold_s"])' "$1" 2>/dev/null; }
field(){ python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["summary"].get(sys.argv[2]))' "$1" "$2" 2>/dev/null; }
park(){ [ -e "$1" ] || return 0; local d="${1%.json}_PARKED_$(date -u +%H%M%SZ).json"; mv "$1" "$d"; echo "    parked $(basename "$d")"; }

attempt(){
  local t=$1 om="$OUT/b2_old_$t.json" nm="$OUT/b2_pairnew_$t.json"
  park "$om"; park "$nm"
  local left=$(( DEADLINE - $(date +%s) )); [ "$left" -gt 600 ] || return 2
  echo "=== $(date -u +%H:%M:%SZ) attempt $t, queueing for the lock (budget ${left}s) ==="
  TAG=$t ENTER_MAX=$ENTER_MAX BENCHLOCK_WAIT_S=$left BENCHLOCK_LOAD_WAIT_S=30 "$BL" pvx-baseline -- bash "$INNER"
  local rc=$?
  [ "$rc" = 0 ] || { echo "    attempt $t: rc=$rc (75=box too loaded, 75/70=lock)"; return 1; }
  summarised "$om" && summarised "$nm" || { echo "    attempt $t: a cell produced no summary"; return 1; }
  local o=$(median "$om") n=$(median "$nm")
  local opb=$(field "$om" partner_busy) npb=$(field "$nm" partner_busy)
  local olm=$(field "$om" load_max_during) nlm=$(field "$nm" load_max_during)
  echo "    old $o s (load_max $olm, partner_busy $opb) | new $n s (load_max $nlm, partner_busy $npb)"
  if [ "$opb" = "True" ] || [ "$npb" = "True" ]; then
    echo "    attempt $t: board partner was drawing power, shared budget -- parking"; park "$om"; park "$nm"; return 1; fi
  awk -v n="$n" -v r="$NEW_REF" -v t="$NEW_TOL" 'BEGIN{d=(n-r)/r; if(d<0)d=-d; exit !(d<=t)}' || {
    echo "    attempt $t: CONTROL FAILED, main arm $n s against $NEW_REF s -- the window was not good"
    park "$om"; park "$nm"; return 1; }
  echo "=== $(date -u +%H:%M:%SZ) PAIR $t ACCEPTED: old $o / new $n = $(awk -v a=$o -v b=$n 'BEGIN{printf "%.4f", a/b}')x, control $(awk -v n=$n -v r=$NEW_REF 'BEGIN{printf "%+.2f", 100*(n-r)/r}') % ==="
  return 0
}

ok=0
for t in s1 s2 s3 s4 s5 s6; do
  attempt "$t"; rc=$?
  [ "$rc" = 2 ] && { echo "=== deadline ==="; break; }
  [ "$rc" = 0 ] && { ok=$((ok+1)); [ "$ok" -ge 2 ] && break; continue; }
  sleep "$BACKOFF"
done
echo "PVXBASELINEP6DONE $(date -u +%FT%TZ) accepted=$ok"
