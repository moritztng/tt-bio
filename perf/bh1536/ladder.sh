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
done
