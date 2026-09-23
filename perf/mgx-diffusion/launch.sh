#!/usr/bin/env bash
# Run a plan on one pinned whglx chip:  launch.sh <card> <plan>
# Log: perf/mgx-diffusion/logs/<plan>-<card>.log, points: perf/mgx-diffusion/runs.jsonl.
set -u
card=$1 plan=$2
cd "$(dirname "$0")/../.."
root=$PWD
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$root RELEASE_GATE_CENSUS_PYTHONPATH=$root
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:mgx-diffusion
export TT_BIO_LEASE_DIR=$HOME/leases TT_METAL_LOGGER_LEVEL=FATAL
export TT_METAL_CACHE=$HOME/.cache/tt-metal-cache-mgxd
export TT_BIO_OPENFOLD3=$HOME/mgxi-weights/of3-p2-155k.pt
export TT_BIO_OPENBIND=$HOME/mgxi-weights/of3-ob-2025-06-30-174k.pt
export MGX_DIFFUSION_WORK=$root/perf/mgx-diffusion/work-$card
log=$root/perf/mgx-diffusion/logs; mkdir -p "$log"
py=$HOME/env/bin/python
name=$(basename "$plan" .txt)-$card
"$py" perf/sizegate/mgx/hold.py "$card" $$ >> "$log/hold-$card.log" 2>&1 &
echo "[$(date -u +%FT%TZ)] START $plan card $card $(git rev-parse --short HEAD)" >> "$log/$name.log"
"$py" perf/mgx-diffusion/ladder.py "$plan" >> "$log/$name.log" 2>&1
rc=$?
echo "[$(date -u +%FT%TZ)] EXIT $rc" >> "$log/$name.log"
