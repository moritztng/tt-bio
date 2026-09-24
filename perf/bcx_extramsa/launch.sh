#!/bin/bash
# bcx-extramsa runs on qb2 card 3, in the environment bcx-seam and bcx-round used.
#   launch.sh <out-name> <script.py> <script args...>
# The script gets --out perf/bcx_extramsa/runs/<out-name>. DUMP=1 adds the optimised-HLO dump
# perf/bcx_seam/hostmap.py needs.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$WT" || exit 1
NAME=$1; SCRIPT=$2; shift 2
OUT=$WT/perf/bcx_extramsa/runs/$NAME
mkdir -p "$OUT"
export TT_VISIBLE_DEVICES=3 TT_BIO_LEASE_CARDS=3 TT_BIO_LEASE_HOLDER=worker:bcx-extramsa
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 PYTHONUNBUFFERED=1
export XLA_FLAGS="--xla_gpu_enable_triton_gemm=false"
if [ "${DUMP:-0}" = 1 ]; then
  XLA_FLAGS="$XLA_FLAGS --xla_dump_to=$OUT/hlo --xla_dump_hlo_as_text --xla_dump_hlo_module_re=.*sequence_design_loss.*"
fi
exec /home/ttuser/bcx_e2e_venv/bin/python "$WT/$SCRIPT" --out "$OUT" "$@"
