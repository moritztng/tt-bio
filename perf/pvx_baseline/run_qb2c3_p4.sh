#!/bin/bash
# Phase 4 — the retry ring, with the two things phase 3 got wrong once benchlock was read properly.
#
# benchlock does NOT abort when its quiet-wait expires: after BENCHLOCK_LOAD_WAIT_S it prints
# "Proceeding, RECORD THIS" and runs anyway. Phase 2 and phase 3 pass 7200, so on a box at
# loadavg 30 (an of3t campaign is running four CPU jobs at 300-450 % each) that means squatting on
# the lock for two hours and then banking a suspect session onto the tag. Two fixes:
#
#   1. Wait for the box to go quiet OUTSIDE the lock, then take the lock with a short quiet-wait.
#      Holding the lock while waiting is what starves the queue, and pvx-orchestrator is queued
#      behind us for a ~3 h campaign.
#   2. A session whose own summary says clean_session=false is renamed _DIRTY_ and retried, so a
#      contaminated run cannot consume the tag. This row already threw one p300c session away for
#      exactly this reason in pass 5; here it is automatic.
set -u
NEW=/home/ttuser/pvx_qb2/new
OLD=/home/ttuser/pvx_qb2/old
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
QUIET_LOAD=${QUIET_LOAD:-4.0}
MAX_DIRTY=${MAX_DIRTY:-2}
DEADLINE=${DEADLINE_EPOCH:-$(( $(date +%s) + 28800 ))}
declare -A DIRTY=()

echo "=== $(date -u +%FT%TZ) phase 4 armed, deadline $(date -u -d @$DEADLINE +%FT%TZ), quiet<=$QUIET_LOAD ==="
WAITPID=${1:-}
if [ -n "$WAITPID" ]; then
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
  echo "=== $(date -u +%H:%M:%SZ) phase-2 chain (pid $WAITPID) exited ==="
fi

keep_harvester() {
  pgrep -f "pvx_qb2/harvest.sh" >/dev/null 2>&1 && return 0
  local left=$(( DEADLINE - $(date +%s) + 1800 ))
  [ "$left" -gt 300 ] || return 0
  echo "=== $(date -u +%H:%M:%SZ) starting a harvester for ${left}s ==="
  ( cd /home/ttuser/.coworker/wt/pvx-baseline && \
    HARVEST_S=$left setsid nohup bash /home/ttuser/pvx_qb2/harvest.sh >>/home/ttuser/pvx_qb2/harvest_p8.log 2>&1 & )
}

# Outside the lock. Returns 1 if the deadline arrives first.
wait_quiet() {
  local l
  while :; do
    [ "$(date +%s)" -lt "$DEADLINE" ] || return 1
    l=$(cut -d" " -f1 /proc/loadavg)
    awk -v a="$l" -v b="$QUIET_LOAD" "BEGIN{exit !(a+0<=b+0)}" && { echo "=== $(date -u +%H:%M:%SZ) box quiet, loadavg $l ==="; return 0; }
    sleep 60
  done
}

summarised() { [ -s "$1" ] && grep -q "\"summary\"" "$1" 2>/dev/null; }
clean()      { grep -q "\"clean_session\": true" "$1" 2>/dev/null; }

run() {
  local tree=$1 tag=$2 model=$3 reps=$4 clk=$5; shift 5
  local f="$OUT/$tag.json"
  while :; do
    summarised "$f" && clean "$f" && { echo "=== $(date -u +%H:%M:%SZ) $tag clean and summarised, done ==="; return 0; }
    if summarised "$f"; then
      local d="$OUT/${tag}_DIRTY_$(date -u +%H%M%SZ).json"
      mv "$f" "$d"; DIRTY[$tag]=$(( ${DIRTY[$tag]:-0} + 1 ))
      echo "=== $(date -u +%H:%M:%SZ) $tag came back co-tenanted, parked as $(basename $d), attempt ${DIRTY[$tag]} of $MAX_DIRTY ==="
      [ "${DIRTY[$tag]}" -ge "$MAX_DIRTY" ] && { echo "=== $tag: out of dirty retries, moving on ==="; return 1; }
    fi
    wait_quiet || { echo "=== $(date -u +%H:%M:%SZ) deadline before $tag could run clean ==="; return 1; }
    local left=$(( DEADLINE - $(date +%s) ))
    [ "$left" -gt 900 ] || { echo "=== past deadline, stopping before $tag ==="; return 1; }
    keep_harvester
    echo "=== $(date -u +%H:%M:%SZ) RUN $tag tree=$tree model=$model reps=$reps clock=$clk budget=${left}s ==="
    BENCHLOCK_WAIT_S=$left BENCHLOCK_LOAD_WAIT_S=900 "$BL" pvx-baseline -- \
      env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
          PYTHONPATH="$tree" \
      "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model "$model" --reps "$reps" --clock "$clk" \
        --out "$f" --tag "$tag" "$@"
    echo "RC=$? $tag"
    keep_harvester
    summarised "$f" || return 1   # benchlock timed out or the fold died; leave it for the next pass
  done
}

run "$OLD" b2_old_s1  boltz2      4 1350
run "$NEW" ptx_new_s1 protenix-v2 4 1350
run "$NEW" b2_new_s2  boltz2      4 1350
run "$OLD" b2_old_s2  boltz2      4 1350
run "$NEW" ptx_new_s2 protenix-v2 4 1350
keep_harvester
echo "PVXBASELINEP4DONE $(date -u +%FT%TZ)"
