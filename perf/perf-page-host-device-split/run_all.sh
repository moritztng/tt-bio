#!/bin/sh
# Five models, one card, each under benchlock. Cheapest fold first so a turn that runs
# short still lands whole models rather than half of the most expensive one.
cd "$(dirname "$0")/../.." || exit 1
CARD=${CARD:-1}
for m in "$@"; do
  echo "########## $m ##########"
  BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-60} \
  ~/.coworker/scripts/benchlock.sh perf-page-host-device-split -- \
    env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD \
        TT_BIO_LEASE_HOLDER=worker:perf-page-host-device-split \
    ~/tt-bio-dev/env/bin/python3 -u perf/perf-page-host-device-split/tt_split.py \
      --model "$m" --rounds 3 \
      --out perf/perf-page-host-device-split/tt_${m}_qb2c${CARD}.json 2>&1
  echo "########## $m rc=$? ##########"
done
