#!/bin/sh
# Which of the three levers moves the coordinates? GLN_ROW_FOLD lives in the confidence head,
# which predicts plDDT/pTM and does not feed the structure, so it should move the scores and
# not the atoms. TRIATT_FUSED_HIFI is in the trunk and should move both. One arm each says so
# instead of assuming it.
cd /home/ttuser/.coworker/wt/rf3-msa-scaling-rootcause || exit 1
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=1
export TT_BIO_LEASE_CARDS=0,1
export TT_BIO_LEASE_HOLDER=worker:rf3-msa-scaling-rootcause
PY=/home/ttuser/tt-bio-dev/env/bin/python3
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|env file|parse_atom_array|element type"
for arm in hifi gln; do
    case $arm in
        hifi) L="TT_BIO_TRIATT_FUSED_HIFI=1" ;;
        gln)  L="TT_BIO_RF3_GLN_ROW_FOLD=1" ;;
    esac
    echo "########## acc298 $arm levers=[$L] start $(date -u +%H:%M:%S)"
    env $L $PY -u perf/rf3/fold_fix_ab.py --fix cdk2x2_298 --label "$arm" \
        --outdir perf/rf3/accuracy/298_$arm --repeat 1 2>&1 | grep -viE "$G"
    echo "########## acc298 $arm done $(date -u +%H:%M:%S)"
done
