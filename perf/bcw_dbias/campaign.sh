#!/bin/bash
# bcw-dbias acceptance: hIL2R + 100 (544 axis), uncapped, through MPNN and validation to the
# campaign's own accept/reject. Arm 1 = looped triatt_bw serving, arm 0 = chunked fallback.
# usage: campaign.sh <arm> <card> <tag>   (tag must be a fresh name)
set -u
arm=$1 card=$2 tag=$3
cd /home/ttuser/.coworker/wt/bcw-dbias
out=perf/bcw_dbias/out; log=$out/campaign.log
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/$out/xlacache
export TT_VISIBLE_DEVICES=$card TT_BIO_LEASE_CARDS=$card TT_BIO_LEASE_HOLDER=worker:bcw-dbias
echo "=== $(date -u +%FT%TZ) $tag arm=$arm card=$card commit=$(git rev-parse --short HEAD)" >> $log
mkdir -p $out/$tag
TT_BIO_TRIATT_BW_FUSED=$arm timeout 14400 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
  --params /home/ttuser/bcx_e2e/af2_params --out $out/$tag --target hIL2R --binder 100 \
  --rounds 0 --trajectories auto --max-trajectories 4 --final-designs 1 > $out/$tag.log 2>&1
echo "=== rc=$? $(date -u +%FT%TZ) $tag" >> $log
