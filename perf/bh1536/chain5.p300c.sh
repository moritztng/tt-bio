#!/usr/bin/env bash
# Re-walk what the wedged card ate, then the queue chains 2-4 never reached. The card was reset
# (tt-smi -r 1) and verified to open before this was launched.
set -u
cd "$(dirname "$0")"
export CARD=1 OUT_TAG=p300c HOLDER=worker:p300c-1536-structure BUDGET=2400
./ladder.sh esmfold2-fast:1536 openfold3:1536 openbind:1536 rf3:1536 opendde-abag:1536 nesso1:1536
./ladder.sh esmfold2:1024 esmfold2:1408 esmfold2:1568 boltz2:1568
./ladder.sh esmfold2:1300 boltz2:1300 protenix-v1:1300 openfold3:1300 rf3:1300 nesso1:1300
FORCE=1 ./ladder.sh esmfold2:1536:rowblock protenix-v1:512:sscontrol protenix-v1:1024:sscontrol
