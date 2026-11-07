#!/bin/bash
# tl.sh OUT CARD TREE [TL_ATTR ARGS...] (on qb1/qb2): triatt_ln stage attribution at OpenDDE's 512 x c384 on one Blackhole card,
# booked through ~/spd/feed/book.py (owner spd-m-opendde-2-c<CARD>-tl), card flock held, AICLK sampled every 0.5 s into OUT/aiclk.tsv.
set -u
OUT=$1 CHIP=$2 T=$3; shift 3
OWN=spd-m-opendde-2-c$CHIP-tl BK=$HOME/spd/feed/book.py SEC=1800
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=${VENV%/}/bin:$PATH
mkdir -p "$OUT"; log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/chain.log"; }
trap 'kill $SAMP 2>/dev/null; python3 $BK rm $OWN; log "chain exit"' EXIT
W=$(python3 $BK add $OWN $CHIP $SEC --note "OpenDDE triatt_ln attribution"); log "book card $CHIP ${SEC}s: $W"
while [ "$(python3 $BK active $CHIP 2>/dev/null)" != "$OWN" ]; do
  sleep 60; W=$(python3 $BK add $OWN $CHIP $SEC --note "OpenDDE triatt_ln attribution")
done
BOX=qb1; [ "$(hostname)" = tt-quietbox2 ] && BOX=qb2
exec 8> "$HOME/spd_${BOX}_chip$CHIP.lock" 9> "$HOME/spd_${BOX}_card$CHIP.lock"
flock -w 1800 8 && flock -w 1800 9 || { log "flock timeout"; exit 3; }
log "window open until ${W#* }, flock held, tree $T, args $*"
( while :; do for n in /sys/class/tenstorrent/tenstorrent\!*; do printf '%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)"; done; sleep 0.5; done ) 8>&- 9>&- >> "$OUT/aiclk.tsv" 2>/dev/null &
SAMP=$!
cd "$T"; export PYTHONPATH=$T TT_VISIBLE_DEVICES=$CHIP
timeout -s INT 1500 python perf/spd_m_protenix/tl_attr.py --n 512 --c 384 "$@" > "$OUT/rows.jsonl" 2> "$OUT/err.log" 8>&- 9>&-
log "tl_attr rc=$? rows=$(wc -l < "$OUT/rows.jsonl")"
