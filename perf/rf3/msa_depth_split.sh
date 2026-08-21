#!/bin/sh
# 2x2: MSA depth (1 = ladder fixture, 35 = perf page fixture) x lever stack (shipped default,
# which is what the published 81.051 s page cell ran, vs the three env levers the pass-9 ladder
# exported for its 25.053 s cell). Same 512 aa sequence, byte-identical query, in all four cells.
cd /home/ttuser/.coworker/wt/rf3-msa-scaling-rootcause || exit 1
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:rf3-msa-scaling-rootcause
export BENCHLOCK_MAXLOAD=${BENCHLOCK_MAXLOAD:-4.0}
export BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-120}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
OUT=perf/rf3/msa_depth
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"
mkdir -p $OUT
for arm in "$@"; do
    case $arm in
        d1_def)   INP=perf/rf3/inputs/rf3_512.json;        LEV=0 ;;
        d35_def)  INP=perf/rf3/inputs/rf3_512_msa35.json;  LEV=0 ;;
        d1_lev)   INP=perf/rf3/inputs/rf3_512.json;        LEV=1 ;;
        d35_lev)  INP=perf/rf3/inputs/rf3_512_msa35.json;  LEV=1 ;;
        *) echo "unknown arm $arm"; exit 1 ;;
    esac
    if [ "$LEV" = 1 ]; then
        L="TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_RF3_GLN_ROW_FOLD=1 TT_BIO_OPM_SMALL_DEPTH=1"
    else
        L=""
    fi
    echo "########## $arm ($INP levers=[$L]) start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    $BL worker:rf3-msa-scaling-rootcause -- env $L $PY -u perf/rf3/msa_decompose.py \
        --aa 512 --input "$INP" --n_recycles 2 \
        --out $OUT/split_512_$arm.json 2>&1 | grep -viE "$G"
    echo "########## $arm done $(date -u +%H:%M:%S)"
done
