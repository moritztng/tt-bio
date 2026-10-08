#!/bin/bash
# lpx-matmul profiler cross-check on the carved .107 chip, after sweep2 has exited (one process on the chip at a time).
cd ~/lpx-matmul
OUT=${1:?out jsonl}; CHIP=${2:?chip}; NODE=${3:?device node}; NTOP=${4:-12}
say(){ echo "$(date -u +%FT%TZ) $*"; }
while pgrep -f "sweep2.py" >/dev/null; do sleep 20; done
grep -q "held by lpx-matmul" ~/lpx/log/keeper-$CHIP.log || { say "chip $CHIP is not carved for lpx-matmul"; exit 3; }
. ~/lpx/census/env.sh      # Tracy build of tt-metal v0.68.0, profiler on; chip, lease and cache overridden below
export TT_VISIBLE_DEVICES=$CHIP LPX_NODE=$NODE TT_BIO_LEASE_HOLDER=lpx-matmul \
       TT_BIO_LEASE_DIR=$HOME/lpx/lpx-matmul/leases TT_BIO_LEASE_TIMEOUT=600 \
       TT_METAL_CACHE=$HOME/lpx-matmul/tracy-cache TT_METAL_LOGS_PATH=$HOME/lpx-matmul/tracy-logs
say "start xcheck load $(cat /proc/loadavg)"
timeout 3600 python xcheck.py r1 "$OUT" "$NTOP"
say "rc=$?"
