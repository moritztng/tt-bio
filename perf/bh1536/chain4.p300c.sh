#!/usr/bin/env bash
# The boundary test. SEQ_LEN_MORE_CHUNKING is 1536 on Blackhole and every gate reading it is a
# strict `>`, so at exactly 1536 the conservative paths are off and at 1568 they are on: a
# bigger fold takes a safer path than a smaller one. 1568 = 49x32, so the token bucket is not a
# variable. Same code as the 1536 rung it is compared against.
set -u
cd "$(dirname "$0")"
WAIT_PID=${WAIT_PID:?}
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) waited out $WAIT_PID, starting chain4"
CARD=1 OUT_TAG=p300c HOLDER=worker:p300c-1536-structure BUDGET=2400 \
  ./ladder.sh esmfold2:1568 boltz2:1568
