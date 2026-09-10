#!/usr/bin/env bash
# Walk the 1536 ladder on ONE card, one rung at a time, skipping rungs already measured.
#
#   CARD=1 OUT_TAG=p300c HOLDER=worker:p300c-1536-structure ./ladder.sh boltz2:1536 rf3:1536
#
# A rung is <model>:<size>[:tag]. `nesso1` is scored as affinity (no structure), everything
# else as predict. The flock is per card and BLOCKING: `flock -n` exits silently and would
# drop the rung instead of waiting for the card (fleet-sh-flock-silent-exit-hides-dispatch).
set -u
cd "$(dirname "$0")"
CARD=${CARD:-0}
OUT_TAG=${OUT_TAG:-}
HOLDER=${HOLDER:-worker:bh-1536}
BUDGET=${BUDGET:-2400}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOCK=/tmp/tt_bio_ladder_card${CARD}.lock
MARK=$(dirname "$0")/.card${CARD}.needs_reset

for rung in "$@"; do
  IFS=: read -r model size tag <<< "$rung"
  if [ -z "${FORCE:-}" ] && ./measured.py "$model" "$size" "$OUT_TAG"; then
    echo "$(date -u +%FT%TZ) skip $model $size (already measured on card $CARD)"
    continue
  fi
  echo "$(date -u +%FT%TZ) rung $model $size card $CARD"
  flock "$LOCK" "$PY" ./run_rung.py --model "$model" --size "$size" --card "$CARD" \
      --holder "$HOLDER" --out_tag "$OUT_TAG" --budget "$BUDGET" \
      ${tag:+--tag "$tag"} \
      $( [ "$model" = nesso1 ] && echo --task affinity )
  echo "$(date -u +%FT%TZ) rung $model $size exit $?"

  # Stop rather than hand the next rung a card the last one broke. A freeze (TIMEOUT) is known to
  # leave the chip failing firmware init, and a wedged chip answers every device open in under a
  # minute -- so one freeze turns the rest of the queue into false ceilings unless the chain looks
  # back. It happened twice on 2026-09-10 (ten rungs, then three). The marker is what a later pass
  # reads to know a reset is owed; clearing the card is deliberately NOT automatic, because
  # `tt-smi -r` here resets the whole board PAIR and the other chip may be a sibling's live job.
  read -r v w klass <<< "$(./last_rung.py "$OUT_TAG")"
  if [ "$klass" = DEVOPEN ]; then
    echo "$(date -u +%FT%TZ) STOP: $model $size never got a working card ($v after ${w}s, its log"\
         "names the device open). Nothing after it would measure anything either." | tee -a "$MARK"
    exit 3
  fi
  if [ "$v" = TIMEOUT ]; then
    echo "$(date -u +%FT%TZ) STOP: $model $size timed out; a freeze wedges this chip, so the rest"         "of the queue would record false ceilings. Reset card $CARD and rerun." | tee -a "$MARK"
    exit 3
  fi
  if [ "$v" = WEDGED ]; then
    fast=$((${fast:-0} + 1))
  else
    fast=0
  fi
  if [ "${fast:-0}" -ge 2 ]; then
    echo "$(date -u +%FT%TZ) STOP: two rungs in a row ended in under a minute ($v), which is the"         "device open failing and not a size. Reset card $CARD and rerun." | tee -a "$MARK"
    exit 3
  fi
done
