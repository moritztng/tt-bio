#!/bin/bash
# One real PD-L1 campaign on qb2, through the user-facing entry, with the same budget on both
# arms. `perf/bcx_p10_campaign/run.sh` is the same thing on pc paths.
#   campaign.sh <tag> <1|auto> [max_trajectories] [extra campaign_run.py args...]
#
# `auto` does not pass `trajectories_per_card` at all, so what runs is the shipped default's
# own resolution on this box. `1` is the old default, BindCraft 2's own loop.
# The compile flock is LIVE: JAX_COMPILATION_CACHE_DIR is set and nothing stubs it out.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; want=$2; budget=${3:-4}; shift 3 || shift 2
out=perf/bcx_default/out/campaign_$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_default/out/xlacache_campaign
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-0} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-0}
export TT_BIO_LEASE_HOLDER=worker:bcx-default
exec /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_campaign/campaign_run.py \
    --trajectories "$want" --max-trajectories "$budget" --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out" "$@"
