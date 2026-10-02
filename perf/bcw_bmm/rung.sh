#!/bin/bash
# One BindCraft 2 rung on qb2 card 0 at the real model: hHSA + binder, rounds gradient rounds.
# usage: rung.sh <tag> <binder> [rounds]
set -u
cd /home/ttuser/.coworker/wt/bcw-bmm
tag=$1 binder=$2 rounds=${3:-1}
out=perf/bcw_bmm/out; mkdir -p $out/$tag
export PYTHONPATH=$PWD BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/$out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcw-bmm
echo "=== $(date -u +%FT%TZ) $tag binder=$binder commit=$(git rev-parse --short HEAD)" >> $out/rungs.log
timeout 2400 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bgx_size/rung.py \
  --params /home/ttuser/bcx_e2e/af2_params --out $out/$tag --target hHSA --binder $binder \
  --rounds $rounds --trajectories 1 > $out/$tag.log 2>&1
echo "=== rc=$? $(date -u +%FT%TZ) $tag" >> $out/rungs.log
