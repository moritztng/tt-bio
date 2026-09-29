#!/bin/bash
# bcx-p10-calls: the composed `hifi` round on pc card 0, so this row's op-name census has its
# own C to reconcile against. `perf/bcx_p10_hostcut/arm_pc.sh` with the route pinned to hifi
# and its own out/ and xla cache, and no other change.
#   arm.sh <tag> <rounds> [extra args...]
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; shift 2
out=perf/bcx_p10_calls/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_calls/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-calls
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u "${ENTRY:-perf/bcx_round/run_round.py}" \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --shipped --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
