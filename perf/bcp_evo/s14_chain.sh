#!/bin/bash
# bcp-evo stage 14 after the wrap fix (3b0501998): float64 stack grade with checkpointing at 128
# and 288, then a flip sitting (2 processes x 22 rounds, ABBA per round), all on card 2.
set -uo pipefail
cd "$(dirname "$0")/../.."
export TT_VISIBLE_DEVICES=2 TT_BIO_LEASE_CARDS=2 TT_BIO_LEASE_HOLDER=worker:bcp-evo PYTHONPATH=$PWD
echo "grade start $(date -u +%FT%TZ)"
for n in 128 288; do
    timeout 2400 ~/bcx_e2e_venv/bin/python3 -u perf/bcp_evo/stack_grade.py --n "$n" \
        --lever tt_bio.inproj_gated:INPROJ_GATED --stats tt_bio.inproj_gated:STATS \
        > "perf/bcp_evo/out/stack_grade_ipg_n$n.log" 2>&1
    echo "grade n=$n rc $? $(date -u +%FT%TZ)"
    grep "== arm" "perf/bcp_evo/out/stack_grade_ipg_n$n.log"
done
base=$(grep "^base=" perf/bcp_evo/s14_sit.sh | sed 's/^base=//; s/"//g')
# shellcheck disable=SC2086
perf/bcp_evo/flip_sit.sh 2 flip_s14c tt_bio.inproj_gated:INPROJ_GATED 2 22 $base TT_BIO_INPROJ_GATED=0
cat perf/bcp_evo/out/flip_s14c/split.txt
echo "chain done $(date -u +%FT%TZ)"
