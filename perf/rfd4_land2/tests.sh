#!/bin/bash
# Card-free sweep of the whole suite, on this branch and on main, under py3.10 / biotite 1.2.0
# and py3.12 / biotite 1.6.0. Device opens are refused (a lease on a card that does not exist),
# so nothing contends with a real card.
# Usage: tests.sh <scratch_root> <branch_tree> <main_tree>
set -u
S=$1; LAND=$2; MAIN=$3
V=$HOME/scratch/shipflow
mkdir -p "$S/tests"
for tree in land main; do
  T=$LAND; [ $tree = main ] && T=$MAIN
  for py in 310 312; do
    P=$V/venv-1384f4c1b7bf/bin/python; [ $py = 312 ] && P=$V/venv312-1384f4c1b7bf/bin/python
    ( cd "$T" && TT_VISIBLE_DEVICES=7 TT_BIO_LEASE_CARDS=7 TT_BIO_LEASE_HOLDER=worker:rfd4-land2 \
      RFD4_BENCHMARKS=$HOME/scratch/rfd4-land2/benchmarks PYTHONPATH="$T" \
      timeout 5400 "$P" -m pytest -q -p no:cacheprovider --junitxml="$S/tests/${tree}${py}.xml" tests \
      ) > "$S/tests/${tree}${py}.log" 2>&1
    echo "$tree$py rc=$?" >> "$S/tests/done"
  done
done
echo FINISHED >> "$S/tests/done"
