#!/bin/bash
# Acceptance with the query loop forced (force_chunk.py), PD-L1 binder 146 at 288 tokens, same
# entry, budget, binder and seed as perf/bcp_land/campaign.sh, whose `on` arm accepted 1 of 6.
#   accept288.sh <card> <tag>   (tag must be a fresh name)
set -u
card=$1 tag=$2
cd /home/ttuser/.coworker/wt/bcw-dbias
out=perf/bcw_dbias/out; log=$out/campaign.log
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/$out/xlacache
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-dbias
echo "=== $(date -u +%FT%TZ) $tag forced-chunk card=$card commit=$(git rev-parse --short HEAD)" >> $log
mkdir -p $out/$tag
timeout 7200 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcw_dbias/force_chunk.py \
  --trajectories auto --max-trajectories 6 --binder 146 \
  --params /home/ttuser/bcx_e2e/af2_params --out $out/$tag > $out/$tag.log 2>&1
echo "=== rc=$? $(date -u +%FT%TZ) $tag" >> $log
