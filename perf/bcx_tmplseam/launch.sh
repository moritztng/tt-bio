#!/bin/bash
# bcx-tmplseam runs on qb2 card 1, in the environment bcx-seam, bcx-round and bcx-extramsa used.
#   launch.sh <out-name> <script.py> <script args...>
# The script gets --out perf/bcx_tmplseam/runs/<out-name>. DUMP=1 adds the optimised-HLO dump
# perf/bcx_seam/hostmap.py needs.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT" || exit 1
NAME=$1; SCRIPT=$2; shift 2
OUT=$WT/perf/bcx_tmplseam/runs/$NAME
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=1 TT_BIO_LEASE_CARDS=1 TT_BIO_LEASE_HOLDER=worker:bcx-tmplseam
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
if [ "${DUMP:-0}" = 1 ]; then
  XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$OUT/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*sequence_design_loss.*"
fi
exec /home/ttuser/bcx_e2e_venv/bin/python -X faulthandler "$WT/$SCRIPT" --out "$OUT" "$@"
