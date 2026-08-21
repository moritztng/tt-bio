#!/usr/bin/env bash
# Two runs, in order, one card. Split because a DRAM-probed run cannot answer a timing or an
# arm-comparison question (see --dram-tags): leg 1 is probe-free and settles the 1024 aa digest,
# plDDT and no-refusal; leg 2 turns the probe on with the per-chunk interior filtered out and
# settles the footprint at both rungs.
set -u
WT=/home/ttuser/.coworker/wt/opendde-size-generality-l1-work-split-p4
PY=/home/ttuser/tt-bio-dev/env/bin/python3
cd $WT || exit 1
export TT_VISIBLE_DEVICES=3
export TT_BIO_LEASE_HOLDER=worker:opendde-size-generality-l1-work-split-p4
export PYTHONPATH=$WT

$PY -u perf/other512/fold_ab_multi.py --model opendde --sizes 1024 --arms on,devcat \
     --out perf/oddel1/p4_1024_qb1c3.json > /tmp/p4_1024.log 2>&1
echo "LEG1 exit $? $(date -u +%FT%TZ)" >> /tmp/p4_chain.log

TT_BIO_DRAM_PEAK=/tmp/p4_dram2.txt $PY -u perf/other512/fold_ab_multi.py --model opendde \
     --sizes 768,1024 --arms on,devcat --dram-tags pairformer \
     --out perf/oddel1/p4_dram_filtered_qb1c3.json > /tmp/p4_dram2.log 2>&1
echo "LEG2 exit $? $(date -u +%FT%TZ)" >> /tmp/p4_chain.log
