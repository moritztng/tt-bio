#!/bin/bash
# trace_ab.sh OUT CARD TREE (on qb1): OpenDDE 512 traced vs untraced diffusion on one p150a, normal then fast, each a
# paired A/B in one process (xmodel_ab.py --ab-trace, 3 warm pairs) at the tree's own BH lever sets, AICLK sampled.
# Books the card through ~/spd/feed/book.py under its own owner, holds the card flock, drops the booking on exit.
set -u
OUT=$1 CHIP=$2 T=$3 BK=$HOME/spd/feed/book.py OWN=spd-m-opendde-2-c$2-trace
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=${VENV%/}/bin:$PATH
mkdir -p "$OUT"; log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/chain.log"; }
trap 'python3 $BK rm $OWN; log "chain exit"' EXIT
log "book card $CHIP: $(python3 $BK add $OWN $CHIP 2700 --note "OpenDDE BH trace A/B" | tail -1)"
while [ "$(python3 $BK active $CHIP 2>/dev/null)" != "$OWN" ]; do sleep 60; python3 $BK add $OWN $CHIP 2700 --note "OpenDDE BH trace A/B" >/dev/null; done
exec 9> "$HOME/spd_qb1_card$CHIP.lock"; flock -w 1800 9 || { log "flock timeout"; exit 3; }
cd "$T"; export PYTHONPATH=$T TT_VISIBLE_DEVICES=$CHIP
for MODE in normal fast; do
  F=; [ $MODE = fast ] && F=--fast
  ( while :; do for n in /sys/class/tenstorrent/tenstorrent\!*; do printf '%s\t%s\t%s\n' "$(date +%s)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)"; done; sleep 0.5; done ) 9>&- >> "$OUT/aiclk-$MODE.tsv" 2>/dev/null &
  S=$!; log "$MODE start"
  timeout -s TERM 1320 timeout -s INT 1200 python perf/of3_4xpd/xmodel_ab.py --model opendde --tree $T --size 512 \
    --repeat 3 --ab-trace --label tr-$MODE $F --out "$OUT/$MODE.json" > "$OUT/$MODE.log" 2>&1 9>&-
  rc=$?; kill $S 2>/dev/null; log "$MODE rc=$rc"
done
log TRACE-AB-DONE
