#!/bin/bash
# Card-free test sweep: every test file that mentions atomworks or RF3, on main's tree and this
# branch, under py3.10 / biotite 1.2.0 and py3.12 / biotite 1.6.0. Device opens are refused
# (lease grant on a card that does not exist) so nothing contends with the gate's card.
# Usage: [TREES="land main"] tests.sh <scratch_root> <main_tree>
set -u
S=$1; MAIN=$2; HERE=$(cd "$(dirname "$0")" && pwd); LAND=$(cd "$HERE/../.." && pwd)
V=$HOME/scratch/shipflow
mkdir -p "$S/tests"
cd "$LAND"
FILES=$(grep -lE "atomworks|rf3" tests/*.py | tr '\n' ' ')
for tree in ${TREES:-land main}; do
  T=$LAND; [ $tree = main ] && T=$MAIN
  for py in 310 312; do
    P=$V/venv-1384f4c1b7bf/bin/python; [ $py = 312 ] && P=$V/venv312-1384f4c1b7bf/bin/python
    ( cd "$T" && TT_VISIBLE_DEVICES=7 TT_BIO_LEASE_CARDS=7 TT_BIO_LEASE_HOLDER=worker:rfd4-land PYTHONPATH="$T" \
      timeout 2400 "$P" -m pytest -q -p no:cacheprovider --junitxml="$S/tests/${tree}${py}.xml" $FILES \
      ) > "$S/tests/${tree}${py}.log" 2>&1
    echo "$tree$py rc=$?" >> "$S/tests/done"
  done
done
