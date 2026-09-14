#!/usr/bin/env bash
# One arm of the op trace, with the same environment run_arm.sh gives a fold.
#
#   trace_arm.sh <VERSION|new|old> <card> <calls> <out.jsonl> [extra op_trace.py args...]
set -euo pipefail
ARM=$1; CARD=$2; CALLS=$3; OUT=$4; shift 4
REPO=$(cd "$(dirname "$0")/../.." && pwd)
case "$ARM" in
  new) PY=/home/ttuser/scratch/ttx/venv-new/bin/python ;;
  old) PY=/home/ttuser/tt-bio-dev/env/bin/python ;;
  *)   PY=/home/ttuser/scratch/ttx/venv-$ARM/bin/python ;;
esac
[ -x "$PY" ] || { echo "no interpreter for arm $ARM at $PY" >&2; exit 2; }
export TT_VISIBLE_DEVICES=$CARD TT_BIO_LEASE_CARDS=$CARD
export TT_BIO_LEASE_HOLDER=worker:ttx-version-recon
export PYTHONPATH=$REPO
export TT_METAL_CACHE=/home/ttuser/scratch/ttx/cache-$ARM
export TT_MESH_GRAPH_DESC_PATH=$REPO/perf/ttx/mgd/bh_1x1.textproto
export TT_BIO_TRIATT_PERSISTENT_MASK=0 TT_BIO_TRIATT_HEAD_MAJOR_QKV=0 \
       TT_BIO_TRIATT_HEAD_MAJOR_TAIL=0 TT_BIO_REBLOCK_PERMUTE_BACK=0 \
       TT_BIO_REBLOCK_PERMUTE_GATED=0 TT_BIO_TRIMUL_DUAL_NOC=0
exec "$PY" "$REPO/perf/ttx/op_trace.py" --out "$OUT" --calls "$CALLS" "$@"
