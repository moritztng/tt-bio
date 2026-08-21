#!/bin/sh
# Pass 9, the four owed batch-1 walls on a QUIET full-grid p150a.
#
# The cells items 1 and 2 want are "the ladder on the unharvested board, measured". qb1 is the
# unharvested board this campaign has been using, and it has now been unavailable for three
# consecutive passes (v0.6.5 release gate + AF2-IG port, loadavg 40-79). pc's card 0 is the SAME
# BOARD: `p9_route_census.py` reads grid (13,10), 130 cores, 1 532 416 B/core, 189.99 MB L1
# budget and the identical L1/DRAM route at all five rungs -- qb1 reads 130 cores and 189.99 MB
# too. So the board question these cells ask is answered here, on a box at loadavg 0.8.
#
# The one thing that is NOT the same is the HOST: 12 cores against qb1's 32. The diffusion
# rollout is host-in-the-loop, so pc can only make a fold LONGER than qb1 would. Every cell
# here is therefore an UPPER BOUND on the qb1 wall, and a rung that clears the bar here clears
# it on qb1 a fortiori. A rung that misses here says nothing about qb1.
cd /home/moritz/.coworker/wt/rf3-perf-p9 || exit 1
export PYTHONPATH=$PWD:/home/moritz/rf3_perf_deps
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_HOLDER=worker:rf3-perf-p9
export TT_BIO_TRIATT_FUSED_HIFI=1
export TT_BIO_RF3_GLN_ROW_FOLD=1
export TT_BIO_OPM_SMALL_DEPTH=1
export BENCHLOCK_MAXLOAD=${BENCHLOCK_MAXLOAD:-4.0}
export BENCHLOCK_LOAD_WAIT_S=${BENCHLOCK_LOAD_WAIT_S:-120}
PY=/home/moritz/tt-bio/env/bin/python3
BL=/home/moritz/.coworker/scripts/benchlock.sh
R=/home/moritz/rf3_perf_work
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|\.env file|parse_atom_array|element type"
aa=${1:?usage: p9_pc.sh <aa> <arms>}
arms=${2:?}
echo "########## PC B1 $aa aa start $(date -u +%H:%M:%S) arms=$arms loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
$BL worker:rf3-perf-p9 -- $PY -u perf/rf3/p4_ladder.py --aa "$aa" \
    --arms "$arms" --reps 2 \
    --ckpt $R/rf3_latest.ckpt --feat_cache $R/featcache \
    --out $R/p9_pc_b1_$aa.json 2>&1 | grep -viE "$G"
echo "########## PC B1 $aa aa done $(date -u +%H:%M:%S) loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
