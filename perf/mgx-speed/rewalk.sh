#!/usr/bin/env bash
# Walk a model again on the same chip once its current walk exits:  rewalk.sh <card> <model> <rungs> <walk pid>
# The chip stays claimed in between, and time_rungs.py resumes, so the pass only re-takes the
# folds that are still missing or were voided by host load.
set -u
card=$1 model=$2 rungs=$3 walk=$4
cd "$(dirname "$0")/../.."
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-speed
"$HOME/env/bin/python" perf/sizegate/mgx/hold.py "$card" $$ >> "perf/mgx-speed/logs/hold-$card.log" 2>&1 &
while kill -0 "$walk" 2>/dev/null; do sleep 20; done
exec bash perf/mgx-speed/launch.sh "$card" "$model" "$rungs"
