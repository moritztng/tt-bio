#!/bin/sh
# One rung of the accurate-softmax trunk A/B, under benchlock, on qb1 card 0.
#
# The effect being measured is sub-1 % (the arithmetic estimate was 0.54 % of an 11.110 s
# 1024 aa recycle), and co-tenant noise on this box is 1-10 %, so the lock is not optional:
# two earlier attempts were abandoned when loadavg spiked to 22+ mid-run. MAXLOAD stays at
# the 2.0 default rather than the 4.0 the p9 ladder used, because 4.0 is already larger than
# the effect.
cd /home/ttuser/.coworker/wt/rf3-trunk-softmax-ab || exit 1
export PYTHONPATH=$PWD:/home/ttuser/rf3_perf_deps
export TT_VISIBLE_DEVICES=0
export TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:rf3-trunk-softmax-ab
PY=/home/ttuser/tt-bio-dev/env/bin/python3
BL=/home/ttuser/.coworker/scripts/benchlock.sh
G="fabric|topology|Environment variable|Cached data not found|DeprecationWarning|ttnn.CONFIG|^Config\{|atomworks|bashrc|current shell|\.env file|parse_atom_array|element type"
aa=${1:?usage: softmax_trunk_run.sh <aa>}
echo "########## trunk A/B $aa aa start $(date -u +%H:%M:%S) loadavg $(cut -d\" \" -f1-3 /proc/loadavg)"
$BL worker:rf3-trunk-softmax-ab -- $PY -u perf/rf3/softmax_trunk_ab.py --aa "$aa" \
    --out perf/rf3/results/softmax_trunk_ab_$aa.json 2>&1 | grep -viE "$G"
echo "########## trunk A/B $aa aa done $(date -u +%H:%M:%S) loadavg $(cut -d\" \" -f1-3 /proc/loadavg)"
