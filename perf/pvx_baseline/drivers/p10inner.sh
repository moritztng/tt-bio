#!/bin/bash
# Phase 10 inner -- the same bracket, aimed at a window a quarter the size.
#
# Phase 9 measured why this row keeps missing: host contention on qb2 is worth 1.85x on a 512 aa
# fold (main reads 14.178 s quiet and 26.163 s at loadavg 19 on 16 cores, both pinned at 1350 MHz
# with zero re-asserts), which is MORE than the AICLK governor's 1.27-1.41x. Phase 9's opening
# control passed at 14.178 s and the window was dead four minutes later. The bracket caught it and
# parked all four cells, which is correct, but the pair needed ~10 minutes of quiet and the box
# gave 3 minutes 22 seconds.
#
# So this phase does not try to hold a longer window. It needs a shorter one. Two changes:
#
#  1. THE OLD ARM GOES FIRST. It is the only cell the p300c ratio cannot be computed without, and
#     phase 9 spent its quiet minutes on the opening control and then lost the old arm two folds
#     from done. Control, old arm, control -- the expensive-and-irreplaceable cell sits in the
#     middle of the bracket rather than after a cell that is merely nice to have.
#  2. PROTENIX IS DROPPED from the critical window. Its denominator is already three p150a
#     sessions spanning 0.18 %, and phase 9 showed its digest is byte-identical across board
#     classes (22bc3eaafd886c17 on both), so the p150a cell carries to the p300c on output
#     identity. A p300c Protenix wall time is worth having and is not worth spending the scarce
#     resource on; take it in a separate, later hold.
#
# Critical window: ~4 minutes of folds instead of ~10.
set -u
TAG=${TAG:?}
OLD=/home/ttuser/pvx_qb2/old
NEW=/home/ttuser/pvx_qb2/new
PY=${PY:-/home/ttuser/tt-bio-dev/env/bin/python3}
OUT=${OUT:-/home/ttuser/pvx_qb2/out10}
CARD=${CARD:-1}
REF=${REF:-14.178}
TOL=${TOL:-0.03}
BRACKET_TOL=${BRACKET_TOL:-0.02}
ENTER_MAX=${ENTER_MAX:-2.0}
SETTLE=${SETTLE:-30}
LOG=${LOG:-/home/ttuser/pvx_qb2/p10_cells.log}
mkdir -p "$OUT"

med(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['summary']['median_fold_s'])" "$1" 2>/dev/null; }
loadmax(){ "$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['summary'].get('load_max_during'))" "$1" 2>/dev/null; }

cell(){
  echo "    $(date -u +%H:%M:%SZ) $4  ($2, $3 timed folds after a cold one, card $CARD)"
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:pvx-baseline \
      PYTHONPATH="$1" \
    "$PY" -u "$1/perf/pvx_baseline/cell.py" --model "$2" --reps "$3" --clock 1350 \
      --out "$OUT/$4.json" --tag "$4" >>"$LOG" 2>&1
}

control_ok(){
  local m; m=$(med "$1")
  [ -n "$m" ] || { echo "    control $2: no summary, treating as a failed window"; return 1; }
  awk -v n="$m" -v r="$REF" -v t="$TOL" -v w="$2" \
    'BEGIN{d=(n-r)/r; if(d<0)d=-d; printf "    control %s: %.3f s against card 1'"'"'s own %.3f s, %+.2f %%\n", w, n, r, 100*(n-r)/r; exit !(d<=t)}'
}

echo "    took the lock at $(date -u +%H:%M:%SZ), loadavg $(cut -d' ' -f1 /proc/loadavg), settling ${SETTLE}s"
sleep "$SETTLE"
l=$(cut -d' ' -f1 /proc/loadavg)
echo "    settled at $(date -u +%H:%M:%SZ), loadavg $l, bar $ENTER_MAX"
awk -v a="$l" -v b="$ENTER_MAX" 'BEGIN{exit !(a+0<=b+0)}' || {
  echo "    loadavg $l over $ENTER_MAX, releasing the lock instead of squatting on it"; exit 75; }

cell "$NEW" boltz2 2 "ctl_open_$TAG" || { echo "    opening control did not run"; exit 1; }
control_ok "$OUT/ctl_open_$TAG.json" open || {
  echo "    window refused BEFORE the old arm, releasing the lock"; exit 76; }

cell "$OLD" boltz2 4 "b2_old_$TAG"  || { echo "    b2_old did not run"; exit 1; }
cell "$NEW" boltz2 2 "ctl_close_$TAG" || { echo "    closing control did not run"; exit 1; }

control_ok "$OUT/ctl_close_$TAG.json" close || {
  echo "    the window degraded during the old arm, parking it"; exit 77; }

o=$(med "$OUT/ctl_open_$TAG.json"); c=$(med "$OUT/ctl_close_$TAG.json")
awk -v a="$o" -v b="$c" -v t="$BRACKET_TOL" \
  'BEGIN{d=(b-a)/a; if(d<0)d=-d; printf "    bracket drift: open %.3f s -> close %.3f s, %+.2f %%\n", a, b, 100*(b-a)/a; exit !(d<=t)}' || {
  echo "    the two controls disagree by more than $BRACKET_TOL, parking the window"; exit 78; }

# The old arm must have been folded at the controls' load, not merely between two good controls.
# Phase 9's usable fold sat at load_mean 1.56 against a control's 0.92, and that gap is the whole
# caveat on its 25.066 s. Refuse a pair whose expensive cell ran hotter than its own bracket.
ol=$(loadmax "$OUT/b2_old_$TAG.json"); cl=$(loadmax "$OUT/ctl_open_$TAG.json")
awk -v a="$ol" -v b="$cl" 'BEGIN{if(a+0 > b+0 + 1.0){printf "    old arm load_max %.2f against the control'"'"'s %.2f, parking\n", a, b; exit 1} printf "    old arm load_max %.2f, control %.2f, matched\n", a, b}' || exit 79

echo "    $(date -u +%H:%M:%SZ) $TAG folded, both controls held, bracket flat, loads matched"
o=$(med "$OUT/b2_old_$TAG.json")
awk -v a="$o" -v b="$c" 'BEGIN{printf "    p300c RATIO: %.3f / %.3f = %.4fx\n", a, b, a/b}'
