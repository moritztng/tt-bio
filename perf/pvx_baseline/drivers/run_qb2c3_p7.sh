#!/bin/bash
# Phase 7 -- Protenix v2 512 aa on the p300c, chained behind phase 6 so two rings never race for
# the lock. These cells were in phase 2's queue and went with it when phase 2 was killed for
# squatting; they put PROTENIX on the board class its retired 54.760 s anchor was actually taken
# on, beside the p150a 52.141 s this row already banked.
#
# Same lock discipline as phase 6: block on the queue, settle 60 s inside the lock because
# /proc/loadavg is a 1-minute average, then decide. What differs is ACCEPTANCE. The boltz2 pair has
# a control with a known answer (the main arm reads 14.360 s on this card), so phase 6 can enter at
# a loose bar and let the control judge. Protenix on a p300c has never been measured, so there is
# no known answer to check against and the proxy is all there is: enter at 3.0, the instrument's
# own HOST_QUIET_MAX, and require clean_session. That bar was unreachable while of3t's bundle_min.py
# held 6.6 cores; it finished at ~04:39Z and the box now idles at loadavg 4 and falling.
set -u
NEW=/home/ttuser/pvx_qb2/new
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
ENTER_MAX=${ENTER_MAX:-3.0}
SETTLE=${SETTLE:-60}
DEADLINE=${DEADLINE_EPOCH:-$(( $(date +%s) + 36000 ))}
mkdir -p "$OUT"
echo "=== $(date -u +%FT%TZ) phase 7, deadline $(date -u -d @$DEADLINE +%FT%TZ), enter<=$ENTER_MAX ==="
WAITPID=${1:-}
if [ -n "$WAITPID" ]; then
  echo "=== waiting on phase 6 (pid $WAITPID) ==="
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 60; done
  echo "=== $(date -u +%H:%M:%SZ) phase 6 exited ==="
fi

summarised(){ [ -s "$1" ] && grep -q '"summary"' "$1" 2>/dev/null; }
clean(){ grep -q '"clean_session": true' "$1" 2>/dev/null; }
park(){ [ -e "$1" ] || return 0; local d="${1%.json}_PARKED_$(date -u +%H%M%SZ).json"; mv "$1" "$d"; echo "    parked $(basename "$d")"; }

cell(){
  local tag=$1 f="$OUT/$tag.json"
  summarised "$f" && clean "$f" && { echo "=== $tag already banked clean ==="; return 0; }
  park "$f"
  local left=$(( DEADLINE - $(date +%s) )); [ "$left" -gt 900 ] || { echo "=== deadline before $tag ==="; return 2; }
  echo "=== $(date -u +%H:%M:%SZ) $tag, queueing for the lock (budget ${left}s) ==="
  BENCHLOCK_WAIT_S=$left BENCHLOCK_LOAD_WAIT_S=120 "$BL" pvx-baseline -- bash -c '
    l=$(cut -d" " -f1 /proc/loadavg)
    echo "    took the lock at $(date -u +%H:%M:%SZ), loadavg $l, settling '"$SETTLE"'s"
    sleep '"$SETTLE"'
    l=$(cut -d" " -f1 /proc/loadavg)
    echo "    settled, loadavg $l, bar '"$ENTER_MAX"'"
    awk -v a="$l" -v b="'"$ENTER_MAX"'" "BEGIN{exit !(a+0<=b+0)}" || {
      echo "    too loaded, releasing the lock"; exit 75; }
    env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
        PYTHONPATH='"$NEW"' '"$PY"' -u '"$NEW"'/perf/pvx_baseline/cell.py \
        --model protenix-v2 --reps 4 --clock 1350 --out '"$OUT"'/'"$tag"'.json --tag '"$tag"'
  ' >>"$OUT/../p7_cells.log" 2>&1
  echo "    RC=$? $tag"
  summarised "$f" || { echo "    $tag produced no summary"; return 1; }
  clean "$f" || { echo "    $tag came back co-tenanted"; park "$f"; return 1; }
  echo "=== $(date -u +%H:%M:%SZ) $tag ACCEPTED, median $(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["summary"]["median_fold_s"])' "$f") s ==="
  return 0
}

for tag in ptx_p300c_s1 ptx_p300c_s2; do
  for try in 1 2 3; do
    cell "$tag"; rc=$?
    [ "$rc" = 0 ] && break
    [ "$rc" = 2 ] && break 2
    sleep 300
  done
done
echo "PVXBASELINEP7DONE $(date -u +%FT%TZ)"
