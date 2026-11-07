#!/bin/bash
# fst2.sh OUT CARD TREE TAP LNW NS (on qb1/qb2): fst2_bench (fused fp32 single Transition vs the composed form) on one
# Blackhole card, booked through ~/spd/feed/book.py (owner spd-m-opendde-2-c<CARD>-fst2), card flock held, AICLK
# sampled every 0.5 s into OUT/aiclk.tsv. The kernel targets Wormhole; on Blackhole this checks it is right.
set -u
OUT=$1 CHIP=$2 T=$3 TAP=$4 LNW=$5 NS=$6
OWN=spd-m-opendde-2-c$CHIP-fst2 BK=$HOME/spd/feed/book.py SEC=900 NOTE="OpenDDE fused fp32 Transition check"
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=${VENV%/}/bin:$PATH
mkdir -p "$OUT"; log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/chain.log"; }
trap 'kill $SAMP 2>/dev/null; python3 $BK rm $OWN; log "chain exit"' EXIT
W=$(python3 $BK add $OWN $CHIP $SEC --note "$NOTE"); log "book card $CHIP ${SEC}s: $W"
while [ "$(python3 $BK active $CHIP 2>/dev/null)" != "$OWN" ]; do
  sleep 60; W=$(python3 $BK add $OWN $CHIP $SEC --note "$NOTE")
done
BOX=qb1; [ "$(hostname)" = tt-quietbox2 ] && BOX=qb2
exec 8> "$HOME/spd_${BOX}_chip$CHIP.lock" 9> "$HOME/spd_${BOX}_card$CHIP.lock"
flock -w 1800 8 && flock -w 1800 9 || { log "flock timeout"; exit 3; }
log "window open until ${W#* }, flock held, tree $T"
( while :; do for n in /sys/class/tenstorrent/tenstorrent\!*; do printf '%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)"; done; sleep 0.5; done ) 8>&- 9>&- >> "$OUT/aiclk.tsv" 2>/dev/null &
SAMP=$!
cd "$T"; export PYTHONPATH=$T TT_VISIBLE_DEVICES=$CHIP
timeout -s INT 600 python perf/spd/opendde_wh/fst2_bench.py "$TAP" "$LNW" "$NS" "$OUT/result.json" > "$OUT/rows.jsonl" 2> "$OUT/err.log" 8>&- 9>&-
log "fst2_bench rc=$? rows=$(wc -l < "$OUT/rows.jsonl")"
