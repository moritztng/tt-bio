#!/usr/bin/env bash
# Second card-1 pass: verify the pair-row-tile fix on the tensor that refused, read this
# board's DRAM shape, and put the controls under protenix-v1's clash_frac.
# Waits for chain1 by PID rather than by pattern: a pgrep on the chain's own name matches the
# wrapper that stays alive with the chain as its child, which is how a sibling's chain3.sh
# waited 19 minutes and ran nothing.
set -u
cd "$(dirname "$0")"
CHAIN1_PID=${CHAIN1_PID:?}
while kill -0 "$CHAIN1_PID" 2>/dev/null; do sleep 30; done
echo "$(date -u +%FT%TZ) chain1 ($CHAIN1_PID) done, starting chain2"
PY=/home/ttuser/tt-bio-dev/env/bin/python3
LOCK=/tmp/tt_bio_ladder_card1.lock
flock "$LOCK" env TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 \
  TT_BIO_LEASE_HOLDER=worker:p300c-1536-structure PYTHONPATH=../.. \
  "$PY" ./card_geometry.py --card 1 --out_tag p300c
echo "$(date -u +%FT%TZ) card_geometry exit $?"
CARD=1 OUT_TAG=p300c HOLDER=worker:p300c-1536-structure BUDGET=2400 FORCE=1 \
  ./ladder.sh esmfold2:1536:rowblock esmfold2-fast:1536:rowblock
CARD=1 OUT_TAG=p300c HOLDER=worker:p300c-1536-structure BUDGET=2400 \
  ./ladder.sh esmfold2:1300 boltz2:1300 protenix-v1:1300 openfold3:1300 rf3:1300 nesso1:1300
CARD=1 OUT_TAG=p300c HOLDER=worker:p300c-1536-structure BUDGET=1800 \
  ./ladder.sh protenix-v1:512:sscontrol protenix-v1:1024:sscontrol
