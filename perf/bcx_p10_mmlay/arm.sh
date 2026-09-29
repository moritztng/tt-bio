#!/bin/bash
# bcx-p10-mmlay leg 4: one composed `hifi` round arm on pc card 0 with TT_BIO_MM_LAYOUT
# set explicitly. `perf/bcx_p10_calls/arm.sh` verbatim -- same cell, same binder, same route --
# with its own out/ and xla cache and the lever as the only difference between two arms.
#   arm.sh <tag> <rounds> <mm_layout 0|1> [extra args...]
# NINE rounds is the ceiling per process: `bcx-p10-rne` found the BindCraft 2 compile path
# deadlocks against itself at round 10 on this configuration. Alternate the arms at the
# process boundary instead of raising it.
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; rounds=$2; mm=$3; shift 3
if [ "$rounds" -gt 9 ]; then echo "arm.sh: 9 rounds is the ceiling, see bcx-p10-rne" >&2; exit 2; fi
out=perf/bcx_p10_mmlay/out/$tag
# BindCraft 2 resumes a campaign from its project folder, so a re-run against a tag that
# already holds trajectory 1 returns in 5 s having measured nothing.
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/moritz/bcx_shipped/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_mmlay/out/xlacache
export TT_VISIBLE_DEVICES=0 TT_BIO_LEASE_CARDS=0 TT_BIO_LEASE_HOLDER=worker:bcx-p10-mmlay
export TT_BIO_MM_LAYOUT=$mm
exec /home/moritz/bcx_hostcut_venv/bin/python3 -u perf/bcx_round/run_round.py \
    --rounds "$rounds" --exact 0 --extra-msa 1 --template 1 \
    --triatt-sdpa 0 --triatt-hifi 1 --shipped --binder 146 \
    --params /home/moritz/bcx_shipped/af2_params --out "$out" "$@"
