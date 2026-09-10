#!/usr/bin/env bash
# esmfold2's ladder BELOW 1536, which neither box has: 1024 is the size the Blackhole single
# pass is kept for, so it is the positive control on the row-tile change (it must still fold,
# unchanged), and 1408 is the first rung above the threshold that is not 1536.
set -u
cd "$(dirname "$0")"
WAIT_PID=${WAIT_PID:?}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) waited out $WAIT_PID, starting chain3"
CARD=1 OUT_TAG=p300c HOLDER=worker:p300c-1536-structure BUDGET=2400 \
  ./ladder.sh esmfold2:1024 esmfold2:1408
