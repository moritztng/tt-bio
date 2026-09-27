#!/bin/bash
# Leg 1 probe on a qb2 card: tritraj's full stack5 configuration, one trajectory, the
# Evoformer taped seam hijacked on round 2 by probe_evo.py.
#   probe_evo.sh <tag> [region MB per bank]
set -euo pipefail
cd "$(dirname "$0")/../.."
tag=$1; region=${2:-256}
out=perf/bcx_p10_trace/out/$tag
rm -rf "$out"; mkdir -p "$out"
export PYTHONPATH=$PWD
export BCX_BC2=/home/ttuser/bcx_e2e/bc2
export JAX_COMPILATION_CACHE_DIR=$PWD/perf/bcx_p10_trace/out/xlacache
export TT_VISIBLE_DEVICES=${TT_VISIBLE_DEVICES:-3} TT_BIO_LEASE_CARDS=${TT_BIO_LEASE_CARDS:-3}
export TT_BIO_LEASE_HOLDER=worker:bcx-p10-trace
export TT_BIO_MM_LAYOUT=1 TT_BIO_TAPED_CHANNEL_MOVE=1 TT_BIO_WIDEN_ADD=1
export TT_BIO_GRAD_FANIN_L1=0 TT_BIO_GENQ_COMPACT=0
export TRACE_PROBE_OUT=$PWD/$out/probe.json TRACE_PROBE_REGION_MB=$region
exec timeout 1500 /home/ttuser/bcx_e2e_venv/bin/python3 -u perf/bcx_p10_trace/probe_evo.py \
    --rounds 3 --interleave 0 --trajectories 1 --binder 146 \
    --params /home/ttuser/bcx_e2e/af2_params --out "$out"
