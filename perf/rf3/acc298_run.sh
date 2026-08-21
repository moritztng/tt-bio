#!/bin/sh
# Accuracy A/B at cdk2x2_298, on the sibling card so it runs beside the timed ladder on card 0.
# No benchlock: these folds are scored on coordinates, not on wall time, so host contention
# cannot change the answer. The fold_s numbers they print are NOT perf numbers.
cd /home/ttuser/.coworker/wt/rf3-msa-scaling-rootcause || exit 1
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_CARDS=0,1
export TT_BIO_LEASE_HOLDER=worker:rf3-msa-scaling-rootcause
PY=/home/ttuser/tt-bio-dev/env/bin/python3
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"
for arm in def lev; do
    if [ "$arm" = lev ]; then
        L="TT_BIO_TRIATT_FUSED_HIFI=1 TT_BIO_RF3_GLN_ROW_FOLD=1 TT_BIO_OPM_SMALL_DEPTH=1"
    else
        L=""
    fi
    echo "########## acc298 $arm levers=[$L] start $(date -u +%H:%M:%S)"
    env $L $PY -u perf/rf3/fold_fix_ab.py --fix cdk2x2_298 --label "$arm" \
        --outdir perf/rf3/accuracy/298_$arm --repeat 2 2>&1 | grep -viE "$G"
    echo "########## acc298 $arm done $(date -u +%H:%M:%S)"
done
