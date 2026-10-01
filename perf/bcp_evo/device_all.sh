#!/bin/bash
# bcp-evo stage 1 on one leased card, detached: kernel bench, float64 block grade, round sitting.
#   setsid nohup perf/bcp_evo/device_all.sh <card> > perf/bcp_evo/out/device_all.log 2>&1 &
# Each step is bounded; a failed step is logged and the chain stops before the round sitting,
# which is only worth a card once the kernel grades.
set -uo pipefail
cd "$(dirname "$0")/../.."
card=$1
o=perf/bcp_evo/out; mkdir -p "$o"
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcp-evo
py=/home/ttuser/bcx_e2e_venv/bin/python3
echo "=== bench $(date -u +%FT%TZ) card $card"
timeout 1800 $py -u perf/bcp_evo/gated_bw_bench.py --reps 20 > "$o/gated_bw_bench.log" 2>&1
rc=$?; echo "bench exit $rc"; [ $rc -eq 0 ] || exit $rc
echo "=== grade $(date -u +%FT%TZ)"
PYTHONPATH=$PWD:/home/ttuser/bcx_e2e/bc2 timeout 5400 $py -u perf/bcp_evo/vjp_grade.py \
    > "$o/vjp_grade.log" 2>&1
rc=$?; echo "grade exit $rc"; [ $rc -eq 0 ] || exit $rc
echo "=== sitting $(date -u +%FT%TZ)"
timeout 14400 perf/bcp_evo/round_sit.sh 9 > "$o/round_sit.log" 2>&1
echo "sitting exit $? $(date -u +%FT%TZ)"
