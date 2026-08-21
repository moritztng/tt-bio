#!/bin/sh
# Pass 9 on qb1 (p150a, 13x10 grid, 130 cores, 189.99 MB L1 budget): the batch-1 ladder MEASURED
# on the unharvested board. `p9_route_census.py` already showed lever 8 is DARK here at every
# rung -- the shipped 1.25 x 144 MB = 180 MB fits the 189.99 MB budget with 5.3 % to spare -- so
# the 768 aa cell carries the p6 arm as a FALSIFICATION of that: it must come back ~1.000x.
cd /home/ttuser/.coworker/wt/rf3-perf-p9 || exit 1
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=${P9_DEV:-0}
export TT_BIO_LEASE_HOLDER=worker:rf3-perf-p9
export TT_BIO_TRIATT_FUSED_HIFI=1
export TT_BIO_RF3_GLN_ROW_FOLD=1
export TT_BIO_OPM_SMALL_DEPTH=1
export BENCHLOCK_MAXLOAD=${BENCHLOCK_MAXLOAD:-4.0}
export BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-120}
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
R=/home/ttuser/rf3_perf_work
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|\.env file|parse_atom_array|element type"
aa=${1:?usage: p9_qb1.sh <aa> <arms>}
arms=${2:?}
echo "########## QB1 B1 $aa aa start $(date -u +%H:%M:%S) arms=$arms loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
$BL worker:rf3-perf-p9 -- $PY -u perf/rf3/p4_ladder.py --aa "$aa" \
    --arms "$arms" --reps 2 --out $R/p9_qb1_b1_$aa.json 2>&1 | grep -viE "$G"
echo "########## QB1 B1 $aa aa done $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
