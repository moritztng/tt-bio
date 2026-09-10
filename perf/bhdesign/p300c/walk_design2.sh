#!/usr/bin/env bash
set -u
cd /home/ttuser/.coworker/wt/p300c-1536-design-embed
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
run() {  # model sizes target
  TT_BIO_LEASE_TIMEOUT=1200 BH_LEASE_CARDS=2,3 $PY perf/bhdesign/ladder.py \
    --model "$1" --sizes "$2" --card 3 --board p300c \
    --holder worker:p300c-1536-design-embed --stop-on-fail --timeout 900 \
    --target "$3" --out "$D/$1.jsonl" --work "$D/work"
}
run boltzgen 256       perf/ceilrfd3/targets/laczc_1008.cif
run rfd3     1536      perf/ceilrfd3/targets/laczc_1008.cif
run pxdesign 1536,1831 perf/bhdesign/targets/big_1831.cif
