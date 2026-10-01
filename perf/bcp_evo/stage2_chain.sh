#!/bin/bash
# bcp-evo stage 2 after its sitting: the float64 stack grade on the same card.
#   setsid nohup perf/bcp_evo/stage2_chain.sh <card> > perf/bcp_evo/out/stage2_chain.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
card=$1
while pgrep -f "perf/bcp_evo/round_sit.sh 9 round_nograd" > /dev/null; do sleep 20; done
echo "=== stack grade $(date -u +%FT%TZ) card $card"
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcp-evo
PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2 BCX_BC2=/home/ttuser/bcx_e2e/bc2 timeout 5400 \
    /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcp_evo/stack_grade.py \
    > perf/bcp_evo/out/stack_grade.log 2>&1
echo "grade exit $? $(date -u +%FT%TZ)"
