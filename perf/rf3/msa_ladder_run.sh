#!/bin/sh
# One process per lever arm: the levers are read at import time. Sizes descending so the
# rungs with the OOM and regression risk run first on a clean device.
cd /home/ttuser/.coworker/wt/rf3-msa-scaling-rootcause || exit 1
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:rf3-msa-scaling-rootcause
export BENCHLOCK_MAXLOAD=${BENCHLOCK_MAXLOAD:-4.0}
export BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-300}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
OUT=perf/rf3/msa_depth
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"
mkdir -p $OUT
for arm in "$@"; do
    if [ "$arm" = lev ]; then
        L="TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_RF3_GLN_ROW_FOLD=1 TT_BIO_OPM_SMALL_DEPTH=1"
    elif [ "$arm" = def ]; then
        L=""
    else
        echo "unknown arm $arm"; exit 1
    fi
    echo "########## ladder $arm levers=[$L] start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    $BL worker:rf3-msa-scaling-rootcause -- env $L $PY -u perf/rf3/msa_ladder.py \
        --sizes 1024,768,512,256,128 --depths 1,35 --n_recycles 2 --arm "$arm" \
        --out $OUT/ladder_$arm.jsonl 2>&1 | grep -viE "$G"
    echo "########## ladder $arm done $(date -u +%H:%M:%S)"
done
