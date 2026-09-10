#!/usr/bin/env bash
# Design-model rungs of the p300c walk, fanned onto the idle sibling chip (card 3) so the
# embedding walk keeps card 2. Both chips sit on the same board, so both are this task s
# for the window and a board-pair reset cannot knock over someone else s run.
set -u
cd /home/ttuser/.coworker/wt/p300c-1536-design-embed
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
run() {  # model sizes target
  TT_BIO_LEASE_TIMEOUT=1200 BH_LEASE_CARDS=2,3 $PY perf/bhdesign/ladder.py \
    --model "$1" --sizes "$2" --card 3 --board p300c \
    --holder worker:p300c-1536-design-embed --stop-on-fail --timeout 1500 \
    --target "$3" --out "$D/$1.jsonl" --work "$D/work"
}
run rfd3     300,1024        perf/ceilrfd3/targets/laczc_1008.cif
run pxdesign 256,512,1008    perf/ceilrfd3/targets/laczc_1008.cif
run boltzgen 256,512         perf/ceilrfd3/targets/laczc_1008.cif
