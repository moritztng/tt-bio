#!/usr/bin/env bash
# Walk a model again on the same chip once its current walk exits:
#   rewalk.sh <card> <model> <rungs> <walk pid> [--refit] [launch.sh extras: threads sigma_reps sigma_rung]
# The chip stays claimed in between, and time_rungs.py resumes, so the pass only re-takes the
# folds that are still missing or were voided by host load. --refit first moves every fold of the
# fit rungs (512-1024) to runs/<model>.refit-<time>.jsonl, so the whole fit is re-taken in one
# sitting while the folds above 1024 stand.
set -u
card=$1 model=$2 rungs=$3 walk=$4 refit=${5:-}; shift 4; [ "$refit" = --refit ] && shift
cd "$(dirname "$0")/../.."
export TT_BIO_LEASE_DIR=$HOME/leases TT_BIO_LEASE_HOLDER=worker:mgx-speed
"$HOME/env/bin/python" perf/sizegate/mgx/hold.py "$card" $$ >> "perf/mgx-speed/logs/hold-$card.log" 2>&1 &
while kill -0 "$walk" 2>/dev/null; do sleep 20; done
if [ "$refit" = --refit ]; then
    runs=perf/mgx-speed/runs/$model.jsonl
    "$HOME/env/bin/python" - "$runs" "${runs%.jsonl}.refit-$(date -u +%Y%m%dT%H%MZ).jsonl" <<'PY'
import json, sys
src, dst = sys.argv[1:]
lines = open(src).read().splitlines()
fit = lambda l: 512 <= json.loads(l)["rung"] <= 1024
open(dst, "w").write("".join(l + "\n" for l in lines if fit(l)))
open(src, "w").write("".join(l + "\n" for l in lines if not fit(l)))
PY
    echo "[$(date -u +%FT%TZ)] REFIT $model card $card: fit folds moved aside" >> "perf/mgx-speed/logs/$model.log"
fi
exec bash perf/mgx-speed/launch.sh "$card" "$model" "$rungs" "$@"
