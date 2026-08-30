#!/usr/bin/env bash
# Binary search the 98-commit first-parent path deb9b307..626be0b5 for the commit that moves
# the OpenDDE 512 aa page fixture's plDDT 0.725246 -> 0.725015. Both endpoints were re-measured
# on this host today and are bit-identical over cold+warm, so the predicate is a clean split at
# 0.72513 (midway between the two clusters), not an equality chase.
#   GOOD = plddt >= 0.72513  (pre-move, 0.725246)
#   BAD  = plddt <  0.72513  (post-move, 0.725015)
# A commit that yields no JSON (the pre-08-27 pair-cond 512 aa deadlock) is SKIPPED to its
# neighbour and the card is reset, per the parent task's 71239e38 wedge.
set -u
WT=/home/moritz/.coworker/wt/opendde-512aa-residual-drift-bisect
PY=/home/moritz/tt-bio/env/bin/python3
OUT=$WT/.bisect-out
LOG=$OUT/search.log
mkdir -p "$OUT"
cd "$WT" || exit 1
mapfile -t FP < perf/bisect512r/fp_list.txt
LO=0        # index into FP of the last known GOOD ( -1 means deb9b307 itself )
HI=$((${#FP[@]}-1))   # index of a known BAD (626be0b5, last element)
: "${THRESH:=0.72513}"
LO=${LO:--1}
HI_OVERRIDE=${HI:-}
declare -A SEEN
[ -n "$HI_OVERRIDE" ] && HI=$HI_OVERRIDE
_read_plddt() {   # $1 = arm json path -> echoes the single plDDT, or empty
  "$PY" -c "
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    vs={f['plddt'] for f in d.get('warm_folds',[]) if f.get('plddt') is not None}
    print(repr(vs.pop()) if len(vs)==1 else '')
except Exception:
    print('')
" "$1" 2>/dev/null
}
measure() {   # $1 = index -> echoes plddt or empty
  local i=$1
  local c=${FP[$i]}
  local s=${c:0:8}
  if [ -n "${SEEN[$s]:-}" ]; then echo "${SEEN[$s]}"; return; fi
  local cached
  cached=$(_read_plddt "$OUT/S_$s.json")
  if [ -n "$cached" ]; then SEEN[$s]=$cached; echo "    (cached) idx=$i $s plddt=$cached" >> "$LOG"; echo "$cached"; return; fi
  echo "--- $(date -u +%FT%TZ) step idx=$i $s $(git log -1 --format=%s $c | cut -c1-70)" >> "$LOG"
  REPEAT=1 "$WT/perf/bisect512r/fold_at.sh" "S_$s" "$c" >> "$OUT/S_$s.log" 2>&1
  local v
  v=$(_read_plddt "$OUT/S_$s.json")
  SEEN[$s]=$v
  echo "    -> plddt='$v'" >> "$LOG"
  echo "$v"
}
while [ $((HI-LO)) -gt 1 ]; do
  MID=$(( (LO+HI)/2 ))
  V=$(measure $MID)
  if [ -z "$V" ]; then
    echo "SKIP idx=$MID ${FP[$MID]:0:8} (no json) -- resetting card" >> "$LOG"
    ~/.local/bin/tt-smi -r 0 >> "$LOG" 2>&1
    # step one commit toward HI and retry; if that also fails, give up on this window
    MID=$((MID+1))
    [ $MID -ge $HI ] && { echo "SKIP-EXHAUSTED window $LO..$HI" >> "$LOG"; break; }
    V=$(measure $MID)
    [ -z "$V" ] && { echo "SKIP-EXHAUSTED window $LO..$HI (two in a row)" >> "$LOG"; break; }
  fi
  if "$PY" -c "import sys; sys.exit(0 if float('$V') < $THRESH else 1)"; then
    HI=$MID; echo "BAD  idx=$MID ${FP[$MID]:0:8} plddt=$V" >> "$LOG"
  else
    LO=$MID; echo "GOOD idx=$MID ${FP[$MID]:0:8} plddt=$V" >> "$LOG"
  fi
  echo "    window now LO=$LO HI=$HI" >> "$LOG"
done
echo "=== FIRST-PARENT CULPRIT: idx=$HI ${FP[$HI]} ===" >> "$LOG"
git log -1 --format='%h %ad %s' --date=short ${FP[$HI]} >> "$LOG"
[ $LO -ge 0 ] && { echo "last good: $(git log -1 --format='%h %ad %s' --date=short ${FP[$LO]})" >> "$LOG"; } || echo "last good: deb9b307 (range start)" >> "$LOG"
git checkout -q wk/opendde-512aa-residual-drift-bisect
