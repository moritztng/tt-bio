#!/bin/bash
# One real BindCraft 2 campaign through `bindcraft2.run_campaign` on pc card 0.
#   run.sh <tag> <trajectories> [extra args]
#
# The compile lock is LIVE: JAX_COMPILATION_CACHE_DIR is set and nothing stubs
# `one_worker_compiles` out, which is the case the nine-round harnesses never ran.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; n=$2; shift 2
out=perf/bcx_p10_campaign/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_campaign/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-campaign
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_p10_campaign/campaign_run.py \
    --trajectories "$n" --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
