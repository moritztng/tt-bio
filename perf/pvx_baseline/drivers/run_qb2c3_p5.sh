#!/bin/bash
# Phase 5 -- the matched p300c pair, with the two defects phase 4 still had.
#
# 1. PHASE 4 ADMITTED AT A LOOSER BAR THAN THE INSTRUMENT ACCEPTS. wait_quiet let a session start
#    at loadavg <= 4.0; cell.py sets host_quiet only at load_max_during <= 3.0 (HOST_QUIET_MAX),
#    sampled DURING. A fold adds ~1-1.5 to the box itself, so entering at 4.0 lands at 5+ and the
#    session is rejected on arrival -- burning one of MAX_DIRTY=2 retries without ever folding
#    clean. Enter at 2.0 instead, below the bar with room for our own load.
#
# 2. THE ARMS WERE RUN AS INDEPENDENT CELLS, SO NOTHING CALIBRATED THE WINDOW. Measured this pass
#    on this card: the SAME new tree, same digest 3fcdf07f23ea72ab, AICLK flat 1350.0/1350 on every
#    fold, reads 14.360 s on a quiet host and 24.023 s at loadavg 17.4 -- 1.673x, with zero device
#    co-tenants, because the contaminant is a pure host-CPU job holding no /dev/tenstorrent fd.
#    A pinned clock does not protect a fold from host contention. So run the pair BACK TO BACK in
#    one window and use the new arm as a control with a known answer: if it does not land near
#    14.360 s the window was not quiet and the old number goes with it, whatever loadavg said.
set -u
NEW=/home/ttuser/pvx_qb2/new
OLD=/home/ttuser/pvx_qb2/old
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/pvx_qb2/out3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
QUIET_LOAD=${QUIET_LOAD:-2.0}
DEADLINE=${DEADLINE_EPOCH:-$(( $(date +%s) + 36000 ))}
NEW_REF=14.360          # b2_new_s1 on this card, quiet, pinned; c14-land-tail reads 14.308 s
NEW_TOL=0.06            # 6 %; the arm is worth 1.673x under load, so this is a wide net
mkdir -p "$OUT"

echo "=== $(date -u +%FT%TZ) phase 5 armed, deadline $(date -u -d @$DEADLINE +%FT%TZ), enter<=$QUIET_LOAD ==="
WAITPID=${1:-}
if [ -n "$WAITPID" ]; then
  while kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
  echo "=== $(date -u +%H:%M:%SZ) prior chain (pid $WAITPID) exited ==="
fi

wait_quiet() {
  local l
  while :; do
    [ "$(date +%s)" -lt "$DEADLINE" ] || return 1
    l=$(cut -d" " -f1 /proc/loadavg)
    awk -v a="$l" -v b="$QUIET_LOAD" "BEGIN{exit !(a+0<=b+0)}" && { echo "=== $(date -u +%H:%M:%SZ) enter, loadavg $l ==="; return 0; }
    sleep 45
  done
}
summarised() { [ -s "$1" ] && grep -q "\"summary\"" "$1" 2>/dev/null; }
clean()      { grep -q "\"clean_session\": true" "$1" 2>/dev/null; }
median()     { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"summary\"][\"median_fold_s\"])" "$1" 2>/dev/null; }
park()       { local d="${1%.json}_DIRTY_$(date -u +%H%M%SZ).json"; mv "$1" "$d"; echo "    parked $(basename "$d")"; }

fold() {   # tree tag model reps -> runs one cell under the lock, no waiting inside it
  local tree=$1 tag=$2 model=$3 reps=$4
  local left=$(( DEADLINE - $(date +%s) ))
  echo "=== $(date -u +%H:%M:%SZ) RUN $tag tree=$tree model=$model reps=$reps ==="
  BENCHLOCK_WAIT_S=$left BENCHLOCK_LOAD_WAIT_S=120 "$BL" pvx-baseline -- \
    env TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
        PYTHONPATH="$tree" \
    "$PY" -u "$tree/perf/pvx_baseline/cell.py" --model "$model" --reps "$reps" --clock 1350 \
      --out "$OUT/$tag.json" --tag "$tag" >>"$OUT/../p5_cells.log" 2>&1
  echo "    RC=$? $tag $(summarised "$OUT/$tag.json" && echo "median=$(median "$OUT/$tag.json") clean=$(clean "$OUT/$tag.json" && echo yes || echo no)")"
}

pair() {   # one matched attempt: old then new, back to back, inside one quiet window
  local n=$1 om="$OUT/b2_old_s$n.json" nm="$OUT/b2_pairnew_s$n.json"
  # A pair that is already banked is never re-folded. Without this a relaunch overwrites a
  # clean session with a fresh attempt that may be worse, losing a good measurement.
  if summarised "$om" && clean "$om" && summarised "$nm" && clean "$nm"; then
    echo "=== $(date -u +%H:%M:%SZ) pair $n already banked clean, keeping it ==="; return 0; fi
  summarised "$om" && clean "$om" || { [ -e "$om" ] && summarised "$om" && park "$om"; }
  [ -e "$nm" ] && summarised "$nm" && park "$nm"
  wait_quiet || { echo "=== $(date -u +%H:%M:%SZ) deadline before attempt $n ==="; return 1; }
  fold "$OLD" "b2_old_s$n"     boltz2 4
  fold "$NEW" "b2_pairnew_s$n" boltz2 4
  local o=$(median "$om") nn=$(median "$nm")
  [ -n "$o" ] && [ -n "$nn" ] || { echo "    attempt $n: a cell did not produce a summary"; return 1; }
  if ! clean "$om" || ! clean "$nm"; then echo "    attempt $n: instrument says co-tenanted"; park "$om"; park "$nm"; return 1; fi
  awk -v n="$nn" -v r="$NEW_REF" -v t="$NEW_TOL" "BEGIN{exit !((n-r)/r<=t && (r-n)/r<=t)}" || {
    echo "    attempt $n: CONTROL FAILED, new arm $nn s against $NEW_REF s reference -- window was not quiet"
    park "$om"; park "$nm"; return 1; }
  echo "=== $(date -u +%H:%M:%SZ) MATCHED PAIR $n: old $o s / new $nn s = $(awk -v a=$o -v b=$nn "BEGIN{printf \"%.4f\", a/b}")x, control within $(awk -v n=$nn -v r=$NEW_REF "BEGIN{printf \"%.2f\", 100*(n-r)/r}") % ==="
  return 0
}

pgrep -f "pvx_qb2/harvest.sh" >/dev/null 2>&1 || \
  ( cd /home/ttuser/.coworker/wt/pvx-baseline && HARVEST_S=$(( DEADLINE - $(date +%s) + 1800 )) \
    setsid nohup bash /home/ttuser/pvx_qb2/harvest.sh >>/home/ttuser/pvx_qb2/harvest_p5b.log 2>&1 & )

for n in 1 2 3 4; do
  pair "$n" && { [ "$n" -ge 2 ] && break; }
  [ "$(date +%s)" -lt "$DEADLINE" ] || break
done
echo "PVXBASELINEP5DONE $(date -u +%FT%TZ)"
