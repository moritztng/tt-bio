#!/bin/bash
# lpx-matmul r2 on the carved .107 chip, after r1's process has exited (one process on the chip at a time).
cd ~/lpx-matmul; . ~/japanfold/env.sh
OUT=${1:?out dir}; CHIP=${2:?chip}; NODE=${3:?device node}; NTOP=${4:-30}
export TT_VISIBLE_DEVICES=$CHIP LPX_NODE=$NODE TT_BIO_LEASE_HOLDER=lpx-matmul \
       TT_BIO_LEASE_DIR=$HOME/lpx/lpx-matmul/leases TT_BIO_LEASE_TIMEOUT=600
say(){ echo "$(date -u +%FT%TZ) $*"; }
while pgrep -f "bench_wh.py r1" >/dev/null; do sleep 20; done
grep -q "held by lpx-matmul" ~/lpx/log/keeper-$CHIP.log || { say "chip $CHIP is not carved for lpx-matmul"; exit 3; }
say "r1 done; start r2 load $(cat /proc/loadavg)"
timeout 10800 python sweep2.py r1 "$OUT" "$NTOP"
say "rc=$?"
