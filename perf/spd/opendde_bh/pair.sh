#!/bin/bash
# pair.sh OUT CARD TREE "NAME=SPEC NAME=SPEC" "PDB PDB ..." SECONDS [LADDER_ARMS] (on qb1 or qb2): grade folds for OpenDDE on
# one Blackhole card (qb1 p150a or qb2 p300c), per complex every arm back to back (bench.py --seed 101 --warm 3, one process each), so whatever finishes
# is paired. Books the card through ~/spd/feed/book.py (owner spd-m-opendde-2-c<CARD>), holds its flock, and stops
# starting work 20 min before the window ends; rerun it to finish the rest (done arm x complex pairs are skipped).
# LADDER_ARMS: an arms file for ladder.sh, run first under the same booking. Grade on pc with perf/spd/grade.py.
set -u
OUT=$1 CHIP=$2 T=$3 ARMS=$4 PDBS=$5 SEC=$6 LADDER=${7:-}
OWN=spd-m-opendde-2-c$CHIP BK=$HOME/spd/feed/book.py
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=${VENV%/}/bin:$PATH
mkdir -p "$OUT"; log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/chain.log"; }
trap 'python3 $BK rm $OWN; log "chain exit"' EXIT
W=$(python3 $BK add $OWN $CHIP $SEC --note "OpenDDE BH grade folds"); log "book card $CHIP ${SEC}s: $W"
while [ "$(python3 $BK active $CHIP 2>/dev/null)" != "$OWN" ]; do
  sleep 60; W=$(python3 $BK add $OWN $CHIP $SEC --note "OpenDDE BH grade folds")
done
END=${W#* }
BOX=qb1; [ "$(hostname)" = tt-quietbox2 ] && BOX=qb2  # runners hold either name, so take both
exec 8> "$HOME/spd_${BOX}_chip$CHIP.lock" 9> "$HOME/spd_${BOX}_card$CHIP.lock"
flock -w 1800 8 && flock -w 1800 9 || { log "flock timeout"; exit 3; }
log "window open until $END, flock held, tree $T"
[ -n "$LADDER" ] && NOBOOK=1 TREE=$T bash "$(dirname "$0")/ladder.sh" "$OUT/ladder" "$CHIP" - "$LADDER" 8>&- 9>&-
cd "$T"; export PYTHONPATH=$T TT_VISIBLE_DEVICES=$CHIP
done_n() { cat "$1" 2>/dev/null | grep -c '"fold_s": [0-9]'; }
for x in $PDBS; do
  for a in $ARMS; do
    name=${a%%=*} spec=${a#*=} d=$OUT/$name/$x
    [ "$(done_n "$d/bench.jsonl")" -ge 4 ] && continue
    [ $(( END - $(date +%s) )) -lt 1200 ] && { log "window closing, stop before $name $x"; exit 0; }
    rm -rf "$d"; mkdir -p "$d"
    timeout -s TERM 2520 timeout -s INT 2400 python perf/spd/bench.py --model opendde --out "$d" --chip "$CHIP" \
      --arm "$name:$spec" --inputs "$x" --seed 101 --warm 3 > "$d.log" 2>&1 8>&- 9>&-
    log "$name $x rc=$? folds=$(done_n "$d/bench.jsonl")"
  done
done
log "PAIR-DONE"
