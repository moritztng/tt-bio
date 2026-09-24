#!/usr/bin/env bash
# Walk models' speed-bar rungs one after another on one pinned whglx chip:
#   launch.sh <card> <model>[,<model>...] <rungs>
# Units: perf/mgx_speed_a4/runs/<model>.jsonl, logs: perf/mgx_speed_a4/logs/<model>.log.
set -u
card=$1 models=$2 rungs=$3
cd "$(dirname "$0")/../.."
export PATH=$HOME/.local/bin:$PATH PYTHONPATH=$PWD
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-speed-a4
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL
# host_threads=2 in every record: an uncapped in-process embedder took 20 cores and 1.5x load
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxa4
log=perf/mgx_speed_a4/logs; mkdir -p "$log"
for m in ${models//,/ }; do
  echo "[$(date -u +%FT%TZ)] START $m card $card rungs $rungs $(git rev-parse --short HEAD)" >> "$log/$m.log"
  "$HOME/env/bin/python" -u perf/mgx_speed_a4/rungs.py "$m" "$rungs" >> "$log/$m.log" 2>&1
  echo "[$(date -u +%FT%TZ)] EXIT $? $m" >> "$log/$m.log"
done
