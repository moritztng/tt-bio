#!/bin/bash
# bc4-bet-overlap step 1 on qb2 chip 3: N=2 (py-spy), N=1 (py-spy), N=2 (clean timing). usage: chain.sh <tag>
set -u
WT=/home/ttuser/.coworker/wt/bc4-bet-overlap
RAW=$HOME/bc4-overlap/$1; mkdir -p "$RAW"; cp "$0" "$RAW/"
LOG=$RAW/chain.log
say() { echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bc4-bet-overlap
export JAPANFOLD_BC2_AF2_WEIGHTS=$HOME/bcx_e2e/af2_params PYTHONPATH=$WT
cd "$WT"
say "start sha=$(git rev-parse HEAD) pid=$$ $(cat /proc/loadavg)"
leg() {  # leg <name> <N> <K> <spy_s>
    say "start $1 N=$2 K=$3 $(cut -d' ' -f1-3 /proc/loadavg)"
    TT_BIO_BC2_RESIDENT_BLOCKS=$3 timeout 2700 $HOME/bcx_e2e_venv/bin/python3 perf/spd/bc2_overlap.py \
        --out "$RAW/$1" --n "$2" --spy "$4" > "$RAW/$1.log" 2>&1
    say "end $1 rc=$? $(cut -d' ' -f1-3 /proc/loadavg)"
}
leg n1q 1 auto 0
leg n2q 2 23 0
leg n1q2 1 auto 0
touch "$RAW/done"; say "chain done"
