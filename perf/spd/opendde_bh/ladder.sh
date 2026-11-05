#!/bin/bash
# ladder.sh OUT CARD SHA9 ARMS_FILE (on qb1): OpenDDE 512 lever ladder on one p150a, one process per arm.
# ARMS_FILE lines: "NAME MODE ADDS" (MODE normal|fast, ADDS comma list joined to the model's own set on this arch, or "-").
# Books the card through ~/spd/feed/book.py, waits for the window, holds the card flock, samples AICLK every 0.5 s.
# TREE=dir runs from that tree instead of the gapboard's (which prunes old trees mid-run); NOBOOK=1 when the caller
# already holds the booking and the flock (pair.sh).
set -u
OUT=$1 CHIP=$2 SHA=$3 ARMS=$4; G=~/spd/gapboard; T=${TREE:-$G/trees/m$SHA}; OWN=spd-m-opendde-2; BK=$HOME/spd/feed/book.py
mkdir -p "$OUT"; log() { echo "$(date -u +%FT%TZ) $*" >> "$OUT/chain.log"; }
VENV=$(ls -dt $HOME/scratch/shipflow/venv-*/done | head -1); VENV=${VENV%done}; export PATH=${VENV%/}/bin:$PATH
export PYTHONPATH=$T TT_VISIBLE_DEVICES=$CHIP
N=$(grep -c . "$ARMS"); SEC=$((N * 600 + 600))
if [ -z "${NOBOOK:-}" ]; then
  trap 'python3 $BK rm $OWN; log "chain exit"' EXIT
  log "book card $CHIP ${SEC}s: $(python3 $BK add $OWN $CHIP $SEC --note "OpenDDE BH lever ladder")"
  while [ "$(python3 $BK active $CHIP 2>/dev/null)" != "$OWN" ]; do sleep 60; python3 $BK add $OWN $CHIP $SEC --note "OpenDDE BH lever ladder" >/dev/null; done
  exec 9> "$HOME/spd_qb1_card$CHIP.lock"; flock -w 3600 9 || { log "flock timeout"; exit 3; }
fi
log "window open, tree $T venv $VENV"
cd $T
while read -r NAME MODE ADDS; do
  [ -z "$NAME" ] || [ "${NAME:0:1}" = "#" ] && continue
  [ -f "$OUT/$NAME/result.json" ] && grep -q median_s "$OUT/$NAME/result.json" && continue
  mkdir -p "$OUT/$NAME"
  F=; [ "$MODE" = fast ] && F=--fast
  L=$(python - "$MODE" "$ADDS" <<'PY' 2>/dev/null | tail -1
import sys, tt_bio.opendde as O, tt_bio.tenstorrent as T
mode, adds = sys.argv[1], sys.argv[2]
base = O.BH_FAST_LEVERS if mode == "fast" else O.BH_NORMAL_LEVERS
print(",".join(sorted(base | (T.parse_levers(adds) if adds != "-" else frozenset()))))
PY
)
  echo "$L" > "$OUT/$NAME/levers.txt"
  ( while :; do l=$(cut -d' ' -f1 /proc/loadavg); for n in /sys/class/tenstorrent/tenstorrent\!*; do printf '%s\t%s\t%s\t%s\n' "$(date +%s.%N)" "${n##*!}" "$(cat $n/tt_aiclk 2>/dev/null)" "$l"; done; sleep 0.5; done ) 9>&- >> "$OUT/$NAME/aiclk.tsv" 2>/dev/null &
  S=$!; log "start $NAME $MODE"
  TT_BIO_LEVERS=$L timeout -s TERM 1620 timeout -s INT 1500 python perf/of3_4xpd/xmodel_ab.py --model opendde --tree $T --size 512 \
    --repeat 3 --label $NAME $F --out "$OUT/$NAME/result.json" > "$OUT/$NAME/harness.log" 2>&1 9>&-
  rc=$?; kill $S 2>/dev/null; log "end $NAME rc=$rc median=$(python3 -c "import json;print(json.load(open('$OUT/$NAME/result.json')).get('median_s'))" 2>/dev/null)"
done < "$ARMS"
log "LADDER-DONE"
