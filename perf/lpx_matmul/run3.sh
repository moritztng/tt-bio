#!/bin/bash
# lpx-matmul r3 then the rank-14 profiler cross-check, on the carved .107 chip, one process at a time.
cd ~/lpx-matmul
OUT=${1:?out dir}; CHIP=${2:?chip}; NODE=${3:?device node}
say(){ echo "$(date -u +%FT%TZ) $*"; }
grep -q "held by lpx-matmul" ~/lpx/log/keeper-$CHIP.log || { say "chip $CHIP is not carved for lpx-matmul"; exit 3; }
LEASE="TT_VISIBLE_DEVICES=$CHIP LPX_NODE=$NODE TT_BIO_LEASE_HOLDER=lpx-matmul TT_BIO_LEASE_DIR=$HOME/lpx/lpx-matmul/leases TT_BIO_LEASE_TIMEOUT=600"
say "start r3 load $(cat /proc/loadavg)"
( . ~/japanfold/env.sh; env $LEASE timeout 7200 python r3.py r1 "$OUT" ${SECTIONS:-pad,gen,chain,opm} )
say "r3 rc=$?"
( . ~/lpx/census/env.sh; env $LEASE TT_METAL_CACHE=$HOME/lpx-matmul/tracy-cache TT_METAL_LOGS_PATH=$HOME/lpx-matmul/tracy-logs \
    timeout 1200 python xcheck.py r1 xcheck.jsonl 99 14 )
say "xcheck14 rc=$?"
