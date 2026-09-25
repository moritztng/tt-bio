#!/usr/bin/env bash
# Hand a chip from a walk on an older candidate to a walk on this one, without killing a fold:
#   retime.sh <card> <model> <rungs> <old launch.sh pid> [old rewalk.sh pid]
# The old walk's time_rungs.py is stopped so it starts no new fold, the fold in flight finishes and
# releases the device itself, then the old walk is ended and this clone's launch.sh takes the chip.
# The chip stays claimed throughout (hold.py), so no other row can take it in the gap.
set -u
card=$1 model=$2 rungs=$3 walk=$4 rewalk=${5:-}
cd "$(dirname "$0")/../.."
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-speed
log=perf/mgx-speed/logs; mkdir -p "$log"
"$HOME/env/bin/python" perf/sizegate/mgx/hold.py "$card" $$ >> "$log/hold-$card.log" 2>&1 &
[ -n "$rewalk" ] && kill "$rewalk" 2>/dev/null
note() { echo "[$(date -u +%FT%TZ)] RETIME $model card $card: $*" >> "$log/$model.log"; }
tr=$(pgrep -P "$walk" -f time_rungs.py | head -1)
if [ -n "$tr" ]; then
    kill -STOP "$tr"
    note "old walk $walk stopped, waiting for its fold in flight"
    fold() { for c in $(pgrep -P "$tr" -f lever_census.py); do
                 [ "$(awk '{print $3}' /proc/$c/stat 2>/dev/null)" != Z ] && return 0; done; return 1; }
    while fold; do sleep 10; done
    kill -KILL "$tr" $(pgrep -P "$tr")
fi
while kill -0 "$walk" 2>/dev/null; do sleep 5; done
note "old walk ended, starting on $(git rev-parse --short HEAD)"
exec bash perf/mgx-speed/launch.sh "$card" "$model" "$rungs"
