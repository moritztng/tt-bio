#!/bin/bash
# lpx-matmul r3 chain section again (layer_norm on a bfp8 input) and the pad cost, after run3.sh has exited.
cd ~/lpx-matmul
OUT=${1:?out dir}; CHIP=${2:?chip}; NODE=${3:?device node}
say(){ echo "$(date -u +%FT%TZ) $*"; }
while pgrep -f "run3.sh" >/dev/null; do sleep 20; done
grep -q "held by lpx-matmul" ~/lpx/log/keeper-$CHIP.log || { say "chip $CHIP is not carved for lpx-matmul"; exit 3; }
say "start r3b load $(cat /proc/loadavg)"
( . ~/japanfold/env.sh; env TT_VISIBLE_DEVICES=$CHIP LPX_NODE=$NODE TT_BIO_LEASE_HOLDER=lpx-matmul \
    TT_BIO_LEASE_DIR=$HOME/lpx/lpx-matmul/leases TT_BIO_LEASE_TIMEOUT=600 timeout 1800 python r3b.py r1 "$OUT" chain,padcost )
say "r3b rc=$?"
