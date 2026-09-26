#!/bin/bash
# bcx-bwbytes, in the environment bcx-seam, bcx-round, bcx-extramsa and bcx-tmplseam used.
#   CARD=<n> launch.sh <out-name> <script.py> <script args...>
# CARD is the UMD logical id, which TT_VISIBLE_DEVICES counts in PCI bus order; on qb1 it is NOT
# the /dev/tenstorrent/N node number, and `perf/bcx_stack/stack.py:sysfs_node` resolves the sysfs
# path from the same ordering so the clock is read off the card that ran.
set -u
: "${CARD:?set CARD to the leased card, e.g. CARD=1 (no default -- this row must not guess one)}"
WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT" || exit 1
NAME=$1; SCRIPT=$2; shift 2
OUT=$WT/perf/bcx_bwbytes/runs/$NAME
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD TT_BIO_LEASE_HOLDER=worker:bcx-bwbytes
export OMP_NUM_THREADS=${OMP:-6} MKL_NUM_THREADS=${OMP:-6} PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
exec /home/ttuser/bcx_e2e_venv/bin/python -X faulthandler "$WT/$SCRIPT" --out "$OUT" "$@"
