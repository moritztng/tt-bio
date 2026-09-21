#!/bin/bash
# allm-model: Boltz-2 through the SAME block census ESMFold2 went through, so the per-call walls
# are comparable. ESMFold2's `PairUpdateBlock` is 36.49 ms/call at 512 aa; the MORE-WORK falsifier
# in the state doc is whether Boltz-2's own trunk block costs the same per call at the same tokens.
# Worktree tree == 47810889f plus this row's perf files, so this is today's engine.
set -u
W=/home/ttuser/.coworker/wt/allm-model
PY=/home/ttuser/tt-bio-dev/env/bin/python3
OUT=/home/ttuser/allm_model/out
BL=/home/ttuser/.coworker/scripts/benchlock.sh
CARD=1
mkdir -p "$OUT"
cd "$W" || exit 1
tag=${1:-b2census_512}
BENCHLOCK_WAIT_S=2400 BENCHLOCK_LOAD_WAIT_S=900 "$BL" allm-model -- \
  env TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:allm-model \
      PYTHONPATH="$W" TT_BIO_AICLK=1350 \
  "$PY" -u "$W/perf/allm_model/blockcensus.py" --model boltz2 --size 512 --card $CARD \
    --folds "cold,A,B:1,B:2,C" --out "$OUT/$tag.json"
rc=$?
echo "RC=$rc $tag"
exit $rc
