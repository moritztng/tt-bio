#!/usr/bin/env bash
# Round 2: push every embedding model until it FAILS, so each row is a measured ceiling and
# not a ladder top. esmc-300m already threw a DRAM oom at 131072, so it gets bisected instead.
set -u
cd /home/ttuser/.coworker/wt/p300c-1536-design-embed
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
run() {
  TT_BIO_LEASE_TIMEOUT=1200 BH_LEASE_CARDS=2,3 $PY perf/bhdesign/ladder.py \
    --model "$1" --sizes "$2" --card 2 --board p300c \
    --holder worker:p300c-1536-design-embed --stop-on-fail --timeout 900 \
    --out "$D/$1.jsonl" --work "$D/work"
}
run esmc-300m   98304
run esmc-600m   98304,131072
run saprot-650m 65536,131072
run saprot-1.3b 65536,131072
run esmc-6b     65536,131072
run saprot-35m  131072
