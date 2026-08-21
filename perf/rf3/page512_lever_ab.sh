#!/bin/sh
# The page cell's own harness, fixture and timed boundary, A/B on the three default-OFF levers
# the pass-9 ladder exported. Arm `def` reproduces the published 81.051 s cell in-session, so the
# lever delta is measured against a live control rather than against a quoted number.
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
REP=${REP:-3}
for arm in "$@"; do
    if [ "$arm" = lev ]; then
        L="TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_RF3_GLN_ROW_FOLD=1 TT_BIO_OPM_SMALL_DEPTH=1"
    else
        L=""
    fi
    echo "########## page512 $arm levers=[$L] start $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    $BL worker:rf3-msa-scaling-rootcause -- env $L $PY -u perf/rf3/page512_tt.py \
        --label "$arm" --repeat "$REP" --out $OUT/page512_$arm.json 2>&1 | grep -viE "$G"
    echo "########## page512 $arm done $(date -u +%H:%M:%S)"
done
