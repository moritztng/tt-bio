#!/bin/bash
# tri512: the standalone tail (main's GROW form) on two snapshot trees, alternated by process, ROUNDS rounds.
#   bash perf/rfd4/tri512/treeab.sh <chip> <outdir> <base tree> <lever tree> [rounds]
set -u
N=$1; R=$2; A=$3; B=$4; K=${5:-3}; mkdir -p $R
export TT_VISIBLE_DEVICES=$N TT_BIO_LEASE_CARDS=$N TT_BIO_LEASE_HOLDER=worker:rfd4-perf-tri512
for k in $(seq $K); do
  for t in $A $B; do
    (cd $t && PYTHONPATH=$t:$t/perf timeout -s INT 240 $HOME/tt-bio-dev/env/bin/python perf/rfd4/g512/tail.py \
      --ns 512,384 --legs "grow=1" --rounds 1 --reps 20 --out $R/$(cat $t/.sha).jsonl > $R/$(cat $t/.sha).log 2>&1) \
      || { echo "tree $t rc=$? round $k"; exit 3; }
  done
done
