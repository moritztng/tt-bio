#!/bin/bash
# Wormhole confirmation on the assembled branch: state/japanfold-wh-cutover.md §5.1, fold rows.
#
# Each row is an independent single-card measurement, so they fan across the free cards
# (UMD 26-30; 31 is persistently wedged and is not touched, and no card is reset -- this
# box is production and shared with a customer). Every card gets a bare-open probe first
# and is quarantined if it throws, because `pick_card` only checks lsof and a wedged card
# looks free.
#
# One arm per row on the assembled tree: §5.1 asks each row to reproduce a number a
# predecessor already measured, not to re-derive a delta. Cold fold discarded, 3 warm.
set -u
TREE=${TREE:-/home/cust-team/mthuening/whbase/wt-whcut}
PY=${PY:-/home/cust-team/mthuening/whbase/tt-bio/env/bin/python3}
OUT=$TREE/perf/whcut/out/wh
mkdir -p "$OUT"
cd "$TREE" || exit 1
echo "WH CHAIN START $(date -u +%FT%TZ) head $(git rev-parse HEAD)"

# model:size:flags -- ordered longest-first so the fan-out does not end with one card
# holding the 1024 aa rows while the others idle.
ROWS="
boltz2:1024:
esmfold2:1024:--fast
boltz2:768:
protenix-v2:640:
boltz2:640:
opendde:512:
esmfold2:512:--fast
protenix-v2:512:
boltz2:512:
boltz2:384:
boltz2:320:
boltz2:298:
boltz2:256:
boltz2:128:
"

CARDS=""
for C in 26 27 28 29 30; do
  if timeout 180 env TT_VISIBLE_DEVICES=$C TT_METAL_LOGGER_LEVEL=FATAL "$PY" -c \
       'import ttnn,sys; d=ttnn.open_device(device_id=0); ttnn.close_device(d)' \
       > "$OUT/probe_$C.log" 2>&1; then
    CARDS="$CARDS $C"; echo "card $C OK"
  else
    echo "card $C QUARANTINED, see probe_$C.log"
  fi
done
[ -n "$CARDS" ] || { echo "no usable card"; exit 1; }
echo "cards:$CARDS"

# Deal the rows round-robin, then run each card's list in one background leg.
i=0
for ROW in $ROWS; do
  set -- $(echo "$ROW" | tr ':' ' ')
  N=$(echo $CARDS | wc -w); K=$(( i % N )); i=$(( i + 1 ))
  C=$(echo $CARDS | cut -d' ' -f$(( K + 1 )))
  echo "$ROW" >> "$OUT/list_$C.txt"
done

for C in $CARDS; do
  (
    while read -r ROW; do
      MODEL=$(echo "$ROW" | cut -d: -f1)
      SIZE=$(echo "$ROW" | cut -d: -f2)
      FLAGS=$(echo "$ROW" | cut -d: -f3)
      LABEL="${MODEL}_${SIZE}${FLAGS:+_fast}"
      echo "=== $LABEL card $C start $(date -u +%H:%M:%S)"
      env TT_VISIBLE_DEVICES=$C TT_METAL_LOGGER_LEVEL=FATAL \
          TT_BIO_LEASE_HOLDER=worker:japanfold-wh-cutover \
        "$PY" perf/of3_4xpd/xmodel_ab.py --model "$MODEL" --tree "$TREE" --size "$SIZE" \
          --repeat 3 --label "$LABEL" $FLAGS --out "$OUT/$LABEL.json" \
          > "$OUT/$LABEL.log" 2>&1
      echo "EXIT $LABEL = $? $(grep -hE 'median' "$OUT/$LABEL.log" | tail -1)"
    done < "$OUT/list_$C.txt"
    echo "CARD $C LEG DONE $(date -u +%FT%TZ)"
  ) > "$OUT/leg_$C.log" 2>&1 &
done
wait
echo "WH CHAIN DONE $(date -u +%FT%TZ)"
