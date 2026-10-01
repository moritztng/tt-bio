#!/bin/bash
# bcp-evo stages 6-9 (qkv_packed, fanin_cast, pair_transpose, gated_packed): float64 block grades on
# one card, earlier stages on and later ones off for each.
#   setsid nohup perf/bcp_evo/stage69_grade.sh <card> <lease cards> > perf/bcp_evo/out/stage69_grade.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
export TT_VISIBLE_DEVICES=$1 TT_BIO_LEASE_CARDS=$2 TT_BIO_LEASE_HOLDER=worker:bcp-evo
for lv in qkv_packed fanin_cast pair_transpose gated_packed; do
    echo "=== block grade $lv $(date -u +%FT%TZ) card $1"
    PYTHONPATH=$PWD timeout 1800 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcp_evo/vjp_grade.py \
        --lever $lv > perf/bcp_evo/out/vjp_$lv.log 2>&1
    echo "grade $lv exit $? $(date -u +%FT%TZ)"
done
