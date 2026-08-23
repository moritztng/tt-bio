#!/bin/sh
# Five models, one card, each under benchlock. Cheapest fold first so a turn that runs
# short still lands whole models rather than half of the most expensive one.
cd "$(dirname "$0")/../.." || exit 1
CARD=${CARD:-1}
# Output tag names the box and card the number came from. It is part of the provenance, not
# decoration: the published cells were all measured on qb2 and nothing in the filename said so.
TAG=${TAG:-qb2c$CARD}
# Lease holder and benchlock label default to this task and are overridable, so another task
# can reuse the harness without impersonating this one to the dispatcher.
HOLDER=${HOLDER:-perf-page-host-device-split}
for m in "$@"; do
  echo "########## $m ##########"
  BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-60} \
  ~/.coworker/scripts/benchlock.sh "$HOLDER" -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
        TT_BIO_LEASE_HOLDER=worker:$HOLDER \
    ~/tt-bio-dev/env/bin/python3 -u perf/perf-page-host-device-split/tt_split.py \
      --model "$m" --rounds 3 \
      --out perf/perf-page-host-device-split/tt_${m}_${TAG}.json 2>&1
  echo "########## $m rc=$? ##########"
done
