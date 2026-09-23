#!/bin/bash
# Queue one model's record on a chip, holding the chip from now until the record ends.
#   queue.sh <card> <model> [pid to wait for]
# Waits on a pid rather than a pgrep pattern: a pattern passed in argv matches this script.
card=$1 model=$2 after=${3:-}
cd "$(dirname "$0")/../../.."
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-instrument
~/env/bin/python perf/sizegate/mgx/hold.py "$card" $$ >> "perf/sizegate/mgx/logs/hold-$card.log" 2>&1 &
[ -n "$after" ] && while kill -0 "$after" 2>/dev/null; do sleep 20; done
exec perf/sizegate/mgx/ladder_card.sh record "$card" "$model"
