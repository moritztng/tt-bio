#!/usr/bin/env bash
# Embedding-model rungs of the p300c walk. Card 2 (this task grant).
set -u
cd /home/ttuser/.coworker/wt/p300c-1536-design-embed
PY=/home/ttuser/tt-bio-dev/env/bin/python3
D=perf/bhdesign/p300c
run() {  # model sizes
  TT_BIO_LEASE_TIMEOUT=1200 BH_LEASE_CARDS=2,3 $PY perf/bhdesign/ladder.py \
    --model "$1" --sizes "$2" --card 2 --board p300c \
    --holder worker:p300c-1536-design-embed --stop-on-fail --timeout 1200 \
    --out "$D/$1.jsonl" --work "$D/work"
}
run esmc-300m   8192,32768,65536,131072
run saprot-35m  1536,2000,8192,32768,65536
run esmc-600m   1536,2000,8192,32768,65536
run saprot-650m 1536,2000,8192,32768
run saprot-1.3b 1536,2000,8192,32768
run esmc-6b     1536,2000,8192,32768
