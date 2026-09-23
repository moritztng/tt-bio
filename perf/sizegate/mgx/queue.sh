#!/bin/bash
# Queue a ladder_card.sh run on a chip, holding the chip from now until the run ends.
#   queue.sh <card> <model> [pid to wait for] [mode, default record] [rungs]
# Waits on a pid rather than a pgrep pattern: a pattern passed in argv matches this script.
card=$1 model=$2 after=${3:-} mode=${4:-record} rungs=${5:-}
cd "$(dirname "$0")/../../.."
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-instrument
~/env/bin/python perf/sizegate/mgx/hold.py "$card" $$ >> "perf/sizegate/mgx/logs/hold-$card.log" 2>&1 &
[ -n "$after" ] && while kill -0 "$after" 2>/dev/null; do sleep 20; done
exec perf/sizegate/mgx/ladder_card.sh "$mode" "$card" "$model" $rungs
