#!/usr/bin/env bash
# Time one model's rungs on one pinned whglx chip:
#   launch.sh <card> <model> <rungs> [threads] [sigma_reps] [sigma_rung]
# Log: perf/mgx_wide_seq/speed/logs/<model>.log, folds: perf/mgx_wide_seq/speed/runs/<model>.jsonl.
set -u
card=$1 model=$2 rungs=$3 threads=${4:-2}; shift $(( $# < 4 ? $# : 4 ))
cd "$(dirname "$0")/../../.."
root=$PWD
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$root RELEASE_GATE_CENSUS_PYTHONPATH=$root
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-wide-seq
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxws
export TT_BIO_OPENFOLD3=$HOME/mgxi-weights/of3-p2-155k.pt
export TT_BIO_OPENBIND=$HOME/mgxi-weights/of3-ob-2025-06-30-174k.pt
export RELEASE_GATE_SIZE_WORKDIR=$root/perf/mgx_wide_seq/speed/work-$model
export RELEASE_GATE_FOLD_TIMEOUT=${RELEASE_GATE_FOLD_TIMEOUT:-5400}
log=$root/perf/mgx_wide_seq/speed/logs; mkdir -p "$log"
py=$HOME/env/bin/python
"$py" perf/mgx_wide_seq/speed/hold.py "$card" $$ >> "$log/hold-$card.log" 2>&1 &
echo "[$(date -u +%FT%TZ)] START $model card $card rungs $rungs threads $threads $(git rev-parse --short HEAD)" >> "$log/$model.log"
"$py" perf/mgx_wide_seq/speed/time_rungs.py "$model" "$rungs" "$threads" "$@" >> "$log/$model.log" 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] EXIT $rc $model" >> "$log/$model.log"
